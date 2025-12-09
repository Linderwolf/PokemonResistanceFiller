# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

from sixslot_model import SixSlotModel

# ---------------------------
# Config / constants
# ---------------------------
TYPE_ORDER = [
    "normal", "fighting", "flying", "poison", "ground", "rock",
    "bug", "ghost", "steel", "fire", "water", "grass",
    "electric", "psychic", "ice", "dragon", "dark", "fairy"
]

TYPE_ICON_PATH = "types"
SPRITE_ICON_PATH = "icons"

# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_resource
def load_model_and_mappings(checkpoint_path="sixslot_best.pt"):
    ck = torch.load(checkpoint_path, map_location="cpu")

    # Expect checkpoint to include "model_state", "id_to_index", "index_to_id"
    index_to_id = ck.get("index_to_id")
    id_to_index = ck.get("id_to_index")

    if index_to_id is None or id_to_index is None:
        # fallback on older key names
        index_to_id = ck.get("index_to_id", None)
        id_to_index = ck.get("id_to_index", None)

    num_classes = len(index_to_id) if index_to_id is not None else len(id_to_index)

    model = SixSlotModel(in_dim=18, hidden=256, num_classes=num_classes)
    model.load_state_dict(ck["model_state"])
    model.eval()

    return model, id_to_index, index_to_id

@st.cache_data
def load_data(defense_csv="defense_vectors_export.csv", pokedex_csv="showdown_pokemon_to_import.csv"):
    df_def = pd.read_csv(defense_csv, index_col=0)
    pokedf = pd.read_csv(pokedex_csv)

    # index pokeids should be ints
    try:
        df_def.index = df_def.index.astype(int)
    except:
        df_def.index = df_def.index.map(int)

    # detect damage columns
    damage_cols = [c for c in df_def.columns if c.endswith("_dmg_taken")]
    if len(damage_cols) != 18:
        st.warning(f"Expected 18 damage columns but found {len(damage_cols)}: {damage_cols}")

    # build defense_vectors dict: pid -> numpy (18,)
    defense_vectors = {
        int(pid): df_def.loc[pid, damage_cols].values.astype(np.float32)
        for pid in df_def.index.unique()
    }

    # build name->id map (lowercase)
    # try to use df_def.name if present, else pokedf
    if "name" in df_def.columns:
        name_series = df_def["name"]
        name_to_id = {str(name).lower(): int(pid) for pid, name in zip(df_def.index, name_series)}
    else:
        name_to_id = {str(name).lower(): int(pid) for pid, name in zip(pokedf["name"], pokedf["pokemon_id"])}

    # also create id->(type1,type2) using pokedf if available
    pid_to_types = {}
    if {"pokemon_id", "type_1"}.issubset(pokedf.columns):
        for _, r in pokedf.iterrows():
            try:
                pid = int(r["pokemon_id"])
            except:
                continue
            t1 = r.get("type_1", None)
            t2 = r.get("type_2", None) if "type_2" in pokedf.columns else None
            pid_to_types[pid] = (str(t1).lower() if pd.notna(t1) else None,
                                 str(t2).lower() if pd.notna(t2) else None)
    return df_def, pokedf, defense_vectors, name_to_id, pid_to_types, damage_cols

# ---------------------------
# Utilities
# ---------------------------
def compute_team_profile_from_damage(team_ids, df_def, damage_cols):
    # df_def indexed by pokemon_id
    rows = df_def.loc[team_ids, damage_cols]  # (5,18)
    # weak if multiplier > 1.0 ; resist if multiplier < 1.0 ; immune = multiplier == 0
    weak_mask = (rows > 1.0).astype(int).sum(axis=0)   # counts per type
    resist_mask = (rows < 1.0).astype(int).sum(axis=0)
    # return dicts ordered by TYPE_ORDER with the column names matching damage_cols order
    # we assume damage_cols order matches TYPE_ORDER order; if not, user should ensure it
    mapping = {t: int(weak_mask[i]) for i, t in enumerate(TYPE_ORDER)}
    mapping_res = {t: int(resist_mask[i]) for i, t in enumerate(TYPE_ORDER)}
    return mapping, mapping_res

## Old
# def draw_type_grid_counts(mapping, label="Counts", icon_path=TYPE_ICON_PATH):
#     st.write(f"**{label}**")
#     cols = st.columns(6)
#     for i, t in enumerate(TYPE_ORDER):
#         with cols[i % 6]:
#             icon_file = os.path.join(icon_path, f"{t}.png")
#             if os.path.exists(icon_file):
#                 st.image(icon_file, width=100)
#             else:
#                 st.write(t.capitalize())

#             st.write(mapping[t])

# Should include colour-mapping
def draw_type_grid_counts(mapping, label="Counts", icon_path=TYPE_ICON_PATH):
    st.write(f"**{label}**")

    is_weak = "weak" in label.lower()
    cols = st.columns(6)

    for i, t in enumerate(TYPE_ORDER):
        count = mapping[t]

        # --- TEXT COLOR LOGIC ---
        if is_weak:
            # WEAKNESSES
            if count >= 4:
                color = "red"
            elif count >= 2:
                color = "yellow"
            else:
                color = "white"
        else:
            # RESISTANCES
            if count >= 3:
                color = "lightgreen"
            elif count >= 1:
                color = "yellow"
            else:
                color = "red"

        icon_file = os.path.join(icon_path, f"{t}.png")

        with cols[i % 6]:
            # No background color — just a clean container
            st.markdown(
                f"""
                <div style="
                    border-radius: 8px;
                    padding: 4px;
                    margin-bottom: 8px;
                    text-align: center;
                ">
                """,
                unsafe_allow_html=True
            )

            # ICON
            if os.path.exists(icon_file):
                st.image(icon_file, width=50)
            else:
                st.write(t.capitalize())

            # COUNT WITH COLORED TEXT
            st.markdown(
                f"""
                <div style="font-size: 20px; margin-top: 4px; font-weight: bold; color: {color};">
                    {count}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def recommend_sixth_local(model, team_ids, defense_vectors, index_to_id, pokedf, top_k=10):
    arr = np.stack([defense_vectors[int(pid)] for pid in team_ids]).astype(np.float32)
    x = torch.tensor(arr).unsqueeze(0)  # (1,5,18)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy().squeeze()

    top_idx = probs.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        pid = int(index_to_id[idx])
        prob = float(probs[idx])

        name = None
        if "pokemon_id" in pokedf.columns:
            match = pokedf.loc[pokedf["pokemon_id"] == pid]
        else:
            match = pokedf.loc[pokedf["id"] == pid]

        if len(match) > 0:
            name = match.iloc[0]["name"]
        else:
            name = f"ID {pid}"

        results.append((pid, name, prob))

    return results

# To-Do::
# Implement sprites
def show_recommendations(recs, df_def, pid_to_types, icon_path=TYPE_ICON_PATH, sprite_path=SPRITE_ICON_PATH):
    st.markdown("Recommended Pokémon")
    for pid, name, prob in recs:
        st.markdown(f"### {name} — {prob*100:.2f}%")

        if sprite_path:
            sprite_file = os.path.join(sprite_path, f"{name.lower()}.png")
            if os.path.exists(sprite_file):
                st.image(sprite_file, width=150)

        types = pid_to_types.get(pid, [])
        type_icons = []

        for t in types:
            icon = os.path.join(icon_path, f"{t.lower()}.png")
            if os.path.exists(icon):
                type_icons.append(f'<img src="{icon}" width="40" style="margin:2px;">')
            else:
                type_icons.append(t.capitalize())


        # # types (if available)
        # t1, t2 = pid_to_types.get(pid, (None, None))
        # if t1 or t2:
        #     cols = st.columns(6)
        #     st.write("Types:")
        #     for i, t in enumerate([t1, t2]):
        #         if t:
        #             with cols[i % 6]:
        #                 icon_file = os.path.join(icon_path, f"{t}.png")
        #                 if os.path.exists(icon_file):
        #                     st.image(icon_file, width=100)
        #                 else:
        #                     st.write(t.capitalize())

        # show that Pokémon's weaknesses / resistances
        # derive weak/resist from df_def damage columns
        if pid in df_def.index:
            row = df_def.loc[pid]
            damage_cols = [c for c in df_def.columns if c.endswith("_dmg_taken")]
            weak = {t: int(row[damage_cols[i]] > 1.0) for i, t in enumerate(TYPE_ORDER)}
            resist = {t: int(row[damage_cols[i]] < 1.0) for i, t in enumerate(TYPE_ORDER)}

            st.write("Weaknesses:")
            cols = st.columns(6)
            for i, t in enumerate(TYPE_ORDER):
                if weak[t]:
                    with cols[i % 6]:
                        icon_file = os.path.join(icon_path, f"{t}.png")
                        if os.path.exists(icon_file):
                            st.image(icon_file, width=100)
                        else:
                            st.write(t.capitalize())

            st.write("Resistances:")
            cols = st.columns(6)
            for i, t in enumerate(TYPE_ORDER):
                if resist[t]:
                    with cols[i % 6]:
                        icon_file = os.path.join(icon_path, f"{t}.png")
                        if os.path.exists(icon_file):
                            st.image(icon_file, width=100)
                        else:
                            st.write(t.capitalize())
        st.markdown("---")

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Pokémon 6th Slot Recommender", layout="wide")
st.title("Pokémon 6th Slot Recommender")

# Load model and data
model, id_to_index, index_to_id = load_model_and_mappings("sixslot_best.pt")
df_def, pokedf, defense_vectors, name_to_id, pid_to_types, damage_cols = load_data("defense_vectors_export.csv", "showdown_pokemon_to_import.csv")

# build sorted list of display names for dropdowns
all_names = sorted(list(name_to_id.keys()))

# columns for input
col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Pokémon 1", all_names, index=0, help="Type or search to find a Pokémon")
    p2 = st.selectbox("Pokémon 2", all_names, index=1)
    p3 = st.selectbox("Pokémon 3", all_names, index=2)
with col2:
    p4 = st.selectbox("Pokémon 4", all_names, index=3)
    p5 = st.selectbox("Pokémon 5", all_names, index=4)

if st.button("Recommend 6th Pokémon"):
    team_names = [p1, p2, p3, p4, p5]
    try:
        team_ids = [int(name_to_id[n.lower()]) for n in team_names]
    except Exception as e:
        st.error(f"Name → ID mapping failed: {e}")
        st.stop()

    # show selected team
    st.subheader("Selected team")
    st.write(", ".join(team_names))

    # compute team profile (weak/resist counts)
    team_weak, team_resist = compute_team_profile_from_damage(team_ids, df_def, damage_cols)

    st.subheader("Team weakness counts")
    draw_type_grid_counts(team_weak, label="Team Weakness Count", icon_path=TYPE_ICON_PATH)

    st.subheader("Team resistance counts")
    draw_type_grid_counts(team_resist, label="Team Resistance Count", icon_path=TYPE_ICON_PATH)

    st.subheader("Recommendations")
    recs = recommend_sixth_local(model, team_ids, defense_vectors, index_to_id, pokedf, top_k=10)
    show_recommendations(recs, df_def, pid_to_types, icon_path=TYPE_ICON_PATH, sprite_path=SPRITE_ICON_PATH)