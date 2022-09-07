from graph_generator.generator import generate_graph


def main(seed=888):

    configs = [
        # dict(name="small", size=50, lateral_connections=5, num_flags=5, add_defense=True, mode="random", mu=0),
        # dict(name="small_no_defense", size=50, lateral_connections=5, num_flags=5, add_defense=False, mode="random", mu=0),
        #    dict(name="medium", size=100, lateral_connections=25, num_flags=10,),
        dict(name="big", size=500, lateral_connections=20, num_flags=20, add_defense=True, mode="random", mu=0),
        dict(name="big_no_defense", size=500, lateral_connections=20, num_flags=20, add_defense=False, mode="random", mu=0),
    ]

    for config in configs:
        generate_graph(**config, seed=seed)


if __name__ == "__main__":
    main()
