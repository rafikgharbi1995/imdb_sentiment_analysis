import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import io

# Téléchargement des ressources NLTK
nltk.download('stopwords')
nltk.download('punkt')


class IMDBReviewClassifier:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_explore_data(self, file_path='IMDB Dataset.csv'):
        """Charge et explore le dataset"""
        st.header("📊 Chargement et Exploration des Données")

        # Chargement des données
        self.df = pd.read_csv(file_path)

        # Affichage des informations de base
        st.subheader("Aperçu du Dataset")
        st.write(f"**Nombre total de critiques :** {len(self.df)}")
        st.write(f"**Colonnes :** {list(self.df.columns)}")

        # Distribution des sentiments
        st.subheader("Distribution des Sentiments")
        sentiment_counts = self.df['sentiment'].value_counts()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Diagramme en barres
        sentiment_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
        ax1.set_title('Distribution des Sentiments')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Nombre de Critiques')

        # Diagramme circulaire
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index,
                autopct='%1.1f%%', colors=['green', 'red'])
        ax2.set_title('Répartition des Sentiments')

        st.pyplot(fig)

        # Longueur des critiques
        self.df['review_length'] = self.df['review'].apply(len)
        st.subheader("Longueur des Critiques")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.df['review_length'], bins=50, alpha=0.7, color='skyblue')
        ax.set_xlabel('Longueur des Critiques')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution de la Longueur des Critiques')
        st.pyplot(fig)

        # Statistiques descriptives
        st.write("**Statistiques descriptives :**")
        st.write(f"- Longueur moyenne : {self.df['review_length'].mean():.2f} caractères")
        st.write(f"- Longueur maximale : {self.df['review_length'].max()} caractères")
        st.write(f"- Longueur minimale : {self.df['review_length'].min()} caractères")

        return self.df

    def preprocess_text(self, text):
        """Prétraite le texte"""
        # Conversion en minuscules
        text = text.lower()

        # Suppression des balises HTML
        text = re.sub(r'<.*?>', '', text)

        # Suppression des URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Suppression de la ponctuation et des caractères spéciaux
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Suppression des stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        return ' '.join(tokens)

    def preprocess_data(self):
        """Prétraite l'ensemble des données"""
        st.header("🔧 Prétraitement des Données")

        # Application du prétraitement
        st.write("Prétraitement des critiques en cours...")
        self.df['cleaned_review'] = self.df['review'].apply(self.preprocess_text)

        # Encodage des labels
        self.df['sentiment_encoded'] = self.df['sentiment'].map({'positive': 1, 'negative': 0})

        # Vectorisation TF-IDF
        st.write("Vectorisation TF-IDF en cours...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(self.df['cleaned_review'])

        # Division des données
        y = self.df['sentiment_encoded']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        st.write(f"**Forme des données d'entraînement :** {self.X_train.shape}")
        st.write(f"**Forme des données de test :** {self.X_test.shape}")

        # Aperçu des caractéristiques TF-IDF
        st.subheader("Caractéristiques TF-IDF")
        feature_names = self.vectorizer.get_feature_names_out()
        st.write(f"**Nombre de caractéristiques :** {len(feature_names)}")
        st.write("**Top 20 caractéristiques :**")
        st.write(feature_names[:20])

        return X, y

    def build_model(self, input_dim):
        """Construit le modèle de réseau neuronal"""
        st.header("🧠 Construction du Modèle")

        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        # Compilation du modèle
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        st.write("**Architecture du modèle :**")
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.text("\n".join(model_summary))

        self.model = model
        return model

    def train_model(self, epochs=10, batch_size=32):
        """Entraîne le modèle"""
        st.header("🏋️‍♂️ Entraînement du Modèle")

        # Callback pour l'arrêt anticipé
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        # Entraînement
        st.write("Début de l'entraînement...")
        history = self.model.fit(
            self.X_train.toarray(), self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Visualisation de l'entraînement
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """Visualise l'historique d'entraînement"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Perte
        ax1.plot(history.history['loss'], label='Perte Entraînement')
        ax1.plot(history.history['val_loss'], label='Perte Validation')
        ax1.set_title('Évolution de la Perte')
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Perte')
        ax1.legend()

        # Précision
        ax2.plot(history.history['accuracy'], label='Précision Entraînement')
        ax2.plot(history.history['val_accuracy'], label='Précision Validation')
        ax2.set_title('Évolution de la Précision')
        ax2.set_xlabel('Époques')
        ax2.set_ylabel('Précision')
        ax2.legend()

        st.pyplot(fig)

    def evaluate_model(self):
        """Évalue le modèle"""
        st.header("📈 Évaluation du Modèle")

        # Évaluation sur les données de test
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            self.X_test.toarray(), self.y_test, verbose=0
        )

        st.subheader("Performance sur les Données de Test")
        st.write(f"**Perte :** {test_loss:.4f}")
        st.write(f"**Précision :** {test_accuracy:.4f}")
        st.write(f"**Précision (metric) :** {test_precision:.4f}")
        st.write(f"**Rappel :** {test_recall:.4f}")

        # Prédictions
        y_pred_proba = self.model.predict(self.X_test.toarray())
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Rapport de classification
        st.subheader("Rapport de Classification")
        report = classification_report(self.y_test, y_pred, target_names=['Negative', 'Positive'])
        st.text(report)

        # Matrice de confusion
        st.subheader("Matrice de Confusion")
        cm = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vraies étiquettes')
        ax.set_title('Matrice de Confusion')
        st.pyplot(fig)

        return test_accuracy

    def predict_sentiment(self, text):
        """Prédit le sentiment d'une nouvelle critique"""
        # Prétraitement
        cleaned_text = self.preprocess_text(text)

        # Vectorisation
        text_vectorized = self.vectorizer.transform([cleaned_text])

        # Prédiction
        prediction_proba = self.model.predict(text_vectorized.toarray())[0][0]
        sentiment = "Positive" if prediction_proba > 0.5 else "Negative"
        confidence = prediction_proba if prediction_proba > 0.5 else 1 - prediction_proba

        return sentiment, confidence, prediction_proba


def main():
    st.set_page_config(
        page_title="Classificateur de Critiques IMDb",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Classificateur de Critiques de Films IMDb")
    st.markdown("""
    Ce système utilise un réseau neuronal pour classer les critiques de films comme **positives** ou **négatives**.
    """)

    # Initialisation du classificateur
    if 'classifier' not in st.session_state:
        st.session_state.classifier = IMDBReviewClassifier()

    classifier = st.session_state.classifier

    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Sélectionnez une section :",
        ["Exploration des Données", "Prétraitement", "Construction du Modèle",
         "Entraînement", "Évaluation", "Prédiction"]
    )

    # Chargement des données
    if classifier.df is None:
        try:
            classifier.load_and_explore_data()
        except FileNotFoundError:
            st.error(
                "Fichier 'IMDB Dataset.csv' non trouvé. Veuillez vous assurer qu'il est dans le répertoire courant.")
            return

    if section == "Exploration des Données":
        st.header("📊 Exploration des Données")

        # Aperçu des données
        st.subheader("Aperçu des Données")
        st.dataframe(classifier.df.head(10))

        # Exemples de critiques
        st.subheader("Exemples de Critiques")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Critique Positive :**")
            positive_review = classifier.df[classifier.df['sentiment'] == 'positive'].iloc[0]['review']
            st.text_area("", positive_review[:500] + "...", height=200)

        with col2:
            st.write("**Critique Négative :**")
            negative_review = classifier.df[classifier.df['sentiment'] == 'negative'].iloc[0]['review']
            st.text_area("", negative_review[:500] + "...", height=200)

    elif section == "Prétraitement":
        classifier.preprocess_data()

        # Aperçu avant/après prétraitement
        st.subheader("Exemple de Prétraitement")
        sample_review = classifier.df.iloc[0]
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Avant prétraitement :**")
            st.text_area("", sample_review['review'][:500] + "...", height=200)

        with col2:
            st.write("**Après prétraitement :**")
            st.text_area("", sample_review['cleaned_review'][:500] + "...", height=200)

    elif section == "Construction du Modèle":
        if classifier.X_train is None:
            st.warning("Veuillez d'abord prétraiter les données dans la section 'Prétraitement'.")
        else:
            classifier.build_model(classifier.X_train.shape[1])

    elif section == "Entraînement":
        if classifier.model is None:
            st.warning("Veuillez d'abord construire le modèle dans la section 'Construction du Modèle'.")
        else:
            st.sidebar.subheader("Paramètres d'Entraînement")
            epochs = st.sidebar.slider("Nombre d'époques", 1, 20, 10)
            batch_size = st.sidebar.slider("Taille du lot", 16, 128, 32)

            if st.button("Démarrer l'Entraînement"):
                with st.spinner("Entraînement en cours..."):
                    classifier.train_model(epochs=epochs, batch_size=batch_size)

    elif section == "Évaluation":
        if classifier.model is None:
            st.warning("Veuillez d'abord entraîner le modèle dans la section 'Entraînement'.")
        else:
            classifier.evaluate_model()

    elif section == "Prédiction":
        st.header("🔮 Prédiction de Sentiment")

        st.markdown("""
        Entrez une critique de film ci-dessous pour prédire si elle est **positive** ou **négative**.
        """)

        # Zone de texte pour la prédiction
        user_review = st.text_area(
            "Votre critique de film :",
            height=200,
            placeholder="Entrez votre critique de film ici..."
        )

        if st.button("Analyser le Sentiment") and user_review:
            with st.spinner("Analyse en cours..."):
                sentiment, confidence, proba = classifier.predict_sentiment(user_review)

                # Affichage des résultats
                st.subheader("Résultat de l'Analyse")

                col1, col2 = st.columns(2)

                with col1:
                    if sentiment == "Positive":
                        st.success(f"**Sentiment : {sentiment}**")
                    else:
                        st.error(f"**Sentiment : {sentiment}**")

                    st.write(f"**Confiance :** {confidence:.2%}")
                    st.write(f"**Score de probabilité :** {proba:.4f}")

                with col2:
                    # Barre de progression
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh(['Sentiment'], [proba * 100], color='green' if sentiment == 'Positive' else 'red')
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Probabilité (%)')
                    ax.set_title('Score de Confiance')
                    st.pyplot(fig)

        # Exemples prédéfinis
        st.subheader("Exemples de Test")
        example_reviews = [
            "This movie was absolutely fantastic! The acting was superb and the plot was engaging from start to finish.",
            "Terrible movie. Poor acting, boring story, and awful direction. I want my money back.",
            "The cinematography was beautiful but the characters were poorly developed and the story was predictable.",
            "An amazing performance by the lead actor. The movie kept me on the edge of my seat throughout."
        ]

        for i, example in enumerate(example_reviews):
            if st.button(f"Tester l'exemple {i + 1}"):
                st.session_state.user_review = example
                st.experimental_rerun()


if __name__ == "__main__":
    main()