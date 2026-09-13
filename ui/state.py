from core.state import new_kb_state
from utils.ids import new_conversation_id

def build_states(gr):
    return (
        gr.State([]),
        gr.State(new_kb_state("single")),
        gr.State([]),
        gr.State(new_conversation_id()),
    )