from graph_generator.generator import generate_graph


def main(seed=888):

    configs = [
        dict(name="small", size=50, lateral_connections=5, num_flags=3,),
        #    dict(name="medium", size=100, lateral_connections=25, num_flags=10,),
        #    dict(name="big", size=200, lateral_connections=50, num_flags=20,),
    ]

    for config in configs:
        generate_graph(**config, seed=seed)


if __name__ == "__main__":
    main()
