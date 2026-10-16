from model import GridConfig


def path_selection_config() -> GridConfig:
    """Reproduce the 10x10 board used to report missing paths in round 29."""
    layout = (
        "S##......#",
        "......###.",
        ".###.#..#.",
        "#..#######",
        "..##.###.#",
        "...######.",
        "##.#.#....",
        "...#..##.#",
        "##.##.#.##",
        "#...##.#.T",
    )
    config = GridConfig.create(rows=10, columns=10, gamma=.9)
    config.apply_policy_to_all([.25, .25, .25, .25, 0.0])
    for i, row in enumerate(layout):
        for j, kind in enumerate(row):
            if kind == "#":
                config.cells[i][j].kind = "forbidden"
                config.cells[i][j].reward = -1.0
    config.cells[9][9].probabilities = [0.0, 0.0, 0.0, 0.0, 1.0]
    config.validate()
    return config
