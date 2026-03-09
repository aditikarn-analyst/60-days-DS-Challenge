#day6
def count_word_frequency(sentence):
    words = sentence.lower().split()

    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1

    return frequency
def main():

    print("=" * 60)
    print("           WORD FREQUENCY ANALYSIS SYSTEM")
    print("=" * 60)
    sentence = input("Enter a sentence: ")
    result = count_word_frequency(sentence)
    print("\n" + "-" * 60)
    print("            WORD FREQUENCY RESULTS")
    print("-" * 60)

    for word, count in result.items():
        print(f"{word:<15} : {count}")

    print("-" * 60)
    print("Analysis completed successfully.")
    print("=" * 60)
if __name__ == "__main__":
    main()