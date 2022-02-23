from nim import train, play

ai = train(10000)
peer_grading_mode = False
for _ in range(5 if peer_grading_mode else 1):
    play(ai, 0)
