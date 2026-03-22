import random

def coin_toss_simulation(n=1000):
    heads = 0

    for _ in range(n):
        if random.choice(["H", "T"]) == "H":
            heads += 1

    probability = heads / n
    return probability


def dice_roll_simulation(n=1000):
    count = 0

    for _ in range(n):
        if random.randint(1, 6) == 6:
            count += 1

    probability = count / n
    return probability


def main():

    print("=" * 60)
    print("        PROBABILITY ANALYSIS (REAL-LIFE SCENARIOS)")
    print("=" * 60)

    prob_heads = coin_toss_simulation()
    print("\n🎲 Coin Toss Simulation:")
    print(f"Probability of getting Heads ≈ {prob_heads:.2f}")

    prob_six = dice_roll_simulation()
    print("\n🎯 Dice Roll Simulation:")
    print(f"Probability of getting 6 ≈ {prob_six:.2f}")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("- Probability of heads ~ 0.5 (fair coin)")
    print("- Probability of rolling 6 ~ 1/6 (~0.17)")
    print("- More trials → more accurate probability")

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()