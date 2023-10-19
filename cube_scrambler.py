import random

def generate_cube_scramble():

    movements = ["U", "U'", "U2", "D", "D'", "D2", "L", "L'", "L2", "R", "R'", "R2", "F", "F'", "F2", "B", "B'", "B2"]
    scramble = []


    while len(scramble) < 24:
        move = random.choice(movements)
        if len(scramble) >= 1 and move == scramble[-1]:
            continue
        scramble.append(move)

    return " ".join(scramble)

if __name__ == "__main__":
    scramble = generate_cube_scramble()
    print("Rubik's Cube Scramble:")
    print(scramble)
