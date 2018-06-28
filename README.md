Code here is disgusting, as should be expected from a work-in-progress project.

The agent has two sources from which it acquires prior knowledge and theory, both located on my hard drive. One is a collection of 500,000 games played by LeelaChessZero, a distributed project that is similar to AlphaZero. Leela's self-play generates many games which are stored as PGN files here: http://lczero.org/training_data. Because Leela's self-play ELO rating is so high, it is useful as opening analysis material and as evaluation data.

I don't have the hardware required to train a policy network, so as a result I've had to cut a couple corners. A lot of corners, actually. However, this project is mostly for fun, though I do hope that the AI eventually becomes powerful enough to defeat a decently strong human opponent.
