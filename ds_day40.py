from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    print("\n" + "=" * 70)
    print("TF-IDF TEXT VECTORIZATION SYSTEM")
    print("=" * 70)

    # Sample text data
    documents = [
        "I love data science and machine learning",
        "Machine learning is amazing",
        "Data science is powerful and useful"
    ]

    print("\nOriginal Documents:")
    for doc in documents:
        print("-", doc)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    print("\nFeature Names:")
    print(feature_names)

    print("\nTF-IDF Matrix:")
    print(tfidf_matrix.toarray())

    print("\nVectorization completed successfully 🚀")


if __name__ == "__main__":
    main()