import pickle
import neat
from flappy_game import Game
import os

# ---- CONFIG ----
NUM_AGENTS = 10
CONFIG_PATH = "config.txt"
BEST_GENOME_PATH = "best_genome.pkl"

def load_best_genome(path):
    with open(path, "rb") as f:
        genome = pickle.load(f)
    return genome

def main():
    # โหลด config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, CONFIG_PATH)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # โหลด genome ที่ดีที่สุด
    best_genome = load_best_genome(BEST_GENOME_PATH)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    score = pop.eval_genomes(best_genome,config)

    # แสดงผลคะแนน
    print("\n[RESULT] Final score:", score)

if __name__ == "__main__":
    main()
