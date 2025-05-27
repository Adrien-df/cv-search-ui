import streamlit as st
import openai
from pinecone import Pinecone

# -------------------------------
# 🔐 Configuration API
# -------------------------------

openai.api_key = st.secrets["OPENAI_API_KEY"]
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

index = pc.Index("quickstart")  # remplace par le nom exact de ton index Pinecone


# -------------------------------
# 🌐 Interface Streamlit
# -------------------------------

st.set_page_config(page_title="IA Léon - Recherche sémantique - Neo2", layout="wide")

# --- 🔹 Header stylisé
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
  <h1 style="margin-bottom: 0;">🔍 Recherche sémantique – Neo2</h1>
  <span style="background-color: #e0e0e0; padding: 6px 12px; border-radius: 8px; font-weight: bold; color: #333;">
    Version 1.0
  </span>
</div>
""", unsafe_allow_html=True)

# --- 📘 Sections sidebar

with st.sidebar.expander("🤖 Qui est Léon ?"):
    st.markdown("""
    <div style="font-size: 13px;">
        <p><strong>Léon</strong> est une IA développée par <strong>Neo2</strong> et alimentée de milliers de dossiers de compétences collectés sur 15 ans.</p>
        <ul>
            <li>Il vous aide à retrouver des informations tirées des expériences professionnelles passées des profils Neo2.</li>
            <li>Il ne connaît <strong>ni les parcours académiques</strong>, <strong>ni les soft skills</strong> des profils.</li>
            <li>Il se base exclusivement sur les expériences professionnelles, qu'elles aient été réalisées via Neo2 ou non.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar.expander("📘 Comment l'utiliser ?"):
    st.markdown("""
    <div style="font-size: 13px;">
        <p><strong>Léon accepte trois types de requêtes :</strong></p>
        <ul>
            <li>Le nom d'une entreprise spécifique</li>
            <li>Un type de compétence en particulier</li>
            <li>Un descriptif plus exhaustif d’un profil recherché</li>
        </ul>
        <p><strong>Plus la requête est riche</strong>, plus les résultats seront pertinents.</p>
        <p>Léon identifie les <strong>3 expériences professionnelles</strong> les plus pertinentes et les explique.</p>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar.expander("🌱 Frugalité"):
    st.markdown("""
    <div style="font-size: 13px;">
        <p><strong>Veuillez utiliser Léon avec parcimonie :</strong></p>
        <ul>
            <li>Chaque requête coûte un peu d'argent</li>
            <li>Et surtout consomme des ressources (eau, énergie pour le refroidissement des serveurs)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# 🎯 Interface principale
# -------------------------------

# --- 🔎 Bloc requête utilisateur
st.markdown("## 🧾 Requête")
query = st.text_input("Formulez votre recherche :", placeholder="Ex.: 'Expérience en conception de tuyauterie industrielle et maîtrise du module LICAD'")

if query:
    with st.spinner("Recherche en cours..."):
        try:
            # Embedding
            response = openai.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            vector = response.data[0].embedding

            # Pinecone query
            results = index.query(
                vector=vector,
                top_k=3,
                include_metadata=True,
                namespace="CV_test200_doc_docx"
            )

            # Prompt construction
            system_prompt = (
                "Tu es un agent qui fait du matching entre des expériences professionnelles "
                "tirées de CV et une requête passée par un utilisateur. Ton objectif est "
                "d'expliquer pourquoi au moins une de ces expériences correspond bien à la requête. Il peut y avoir trois types de requêtes. Soit un descriptif d'un profil, soit le nom d'une entreprise particulière soit un type de compétence particulier. Soit nuancé et capable d'évaluer la pertinence du matching. "
            )

            experiences_text = ""
            for i, match in enumerate(results["matches"], start=1):
                meta = match.get("metadata", {})
                experiences_text += f"Expérience {i}:\n"
                for key, value in meta.items():
                    experiences_text += f"{key}: {value}\n"
                experiences_text += "\n"

            user_message = f"Requête : {query}\n\nExpériences :\n{experiences_text.strip()}"

            # LLM Call
            chat_completion = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2
            )

            # --- 🧠 Bloc LLM
            st.markdown("## 🧠 Interprétation de Léon")
            st.markdown(f"""
            <div style="border: 1px solid #DDD; padding: 15px; border-radius: 8px; background-color: #d6d7f2;color: #333;">
            {chat_completion.choices[0].message.content}
            </div>
            """, unsafe_allow_html=True)

            # --- 📄 Bloc expériences
            st.markdown("## 📄 Expériences similaires trouvées")
            for i, match in enumerate(results["matches"], start=1):
                meta = match.get("metadata", {})
                score = round(match.get("score", 0), 3)

                entreprise = meta.get("entreprise", "Entreprise inconnue")
                fonction = meta.get("poste", "Fonction inconnue")
                ref = meta.get("source", "Réf. inconnue")
                duree = meta.get("duree_mois", "Durée inconnue")

                title = f"🔹 EXPERIENCE {i} : {entreprise} | {fonction} | {duree}  | {ref}| Score: {score}"
                with st.expander(title):
                    for key, value in meta.items():
                        st.markdown(f"**{key}** : {value}")
        except Exception as e:
            st.error(f"Erreur lors de la recherche : {e}")

        # --- 📝 Option de follow-up : rédaction d’un mémo
    st.markdown("## 📝 Mémo personnalisé")
    st.markdown("**Voulez-vous que je rédige un mémo commercial prêt à envoyer au client pour convaincre que nous avons les bons profils référencés chez Neo2 par rapport à la requête?**")

    # Boutons pour chaque expérience
    col1, col2, col3 = st.columns(3)
    memo_requested = None

    with col1:
        if st.button("🧑‍💼 Expérience 1"):
            memo_requested = 0
    with col2:
        if st.button("👨‍💼 Expérience 2"):
            memo_requested = 1
    with col3:
        if st.button("👩‍💼 Expérience 3"):
            memo_requested = 2

    if memo_requested is not None:
        try:
            selected_match = results["matches"][memo_requested]
            selected_meta = selected_match.get("metadata", {})
            
            # Génère le descriptif complet à partir des métadonnées
            experience_description = ""
            for key, value in selected_meta.items():
                experience_description += f"{key}: {value}\n"

            # Placeholder du system prompt (modifiable)
            memo_system_prompt = (
                "Tu es un agent qui incarne un chasseur de tête. Tu as pour objectif de rédiger un mémo qui justife pourquoi une requête d'un client correspond à une expérience professionnelle choisie. On va te fournir une requête qui peut être soit un descriptif d'un profil professionnel recherché, soit le nom d'une entreprise particulière soit un type de compétence particulier. Et on va te fournir également le descriptif d'une expérience professionnelle qui, à priori, correspond à la requête. Tu dois argumentr et justifier de la pertinence du matching, tout en gardant un ton factuel et professionnel. Tu te contentes de rédiger le mémo. La réponse doit faire 200 mots maximum. Tu commences avec ce début de phrase 'Parmis nos profils référencés, nous en avons un qui correspond particulièrement bien à votre requête. En effet [...] '"
                
            )

            user_input_for_memo = f"Requête : {query}\n\nExpérience :\n{experience_description.strip()}"

            with st.spinner("✍️ Rédaction du mémo en cours..."):
                memo_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": memo_system_prompt},
                        {"role": "user", "content": user_input_for_memo}
                    ],
                    temperature=0.3
                )

            memo_text = memo_response.choices[0].message.content

            st.markdown("### ✍️ Mémo généré :")
            st.text_area("Mémo", memo_text, height=300)

        except Exception as e:
            st.error(f"Erreur lors de la génération du mémo : {e}")
