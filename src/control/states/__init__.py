"""Recon-lap FSM states.

Each state owns its own file. Transitions are encoded by states returning
the next State instance from `tick()`. To avoid import cycles between
states that point at each other (search <-> approach, pass_through ->
search), follow-on states are imported lazily inside the methods that
return them, not at module load.
"""
