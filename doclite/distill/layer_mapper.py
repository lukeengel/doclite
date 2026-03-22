"""
Layer mapping utilities for teacher-student distillation.

THEORY:
Teacher and student can have different depths.
So we map student layers to teacher layers uniformly.
"""

def build_layer_map(num_teacher_layers: int, num_student_layers: int):
    """
    Build a uniform mapping from student layers to teacher layers.

    Args:
        num_teacher_layers: number of teacher transformer layers
        num_student_layers: number of student transformer layers

    Returns:
        list of teacher layer indices, one for each student layer
    """

    if num_teacher_layers < num_student_layers:
        raise ValueError(
            f"Teacher has fewer layers ({num_teacher_layers}) than student ({num_student_layers})."
        )
    
    if num_student_layers <= 0:
        raise ValueError("num_student_layers must be > 0")
    
    # core logic of mapping teacher layers to student layers through indices
    mapping = []
    for i in range(num_student_layers):
        teacher_idx = int((i+1)* num_teacher_layers/num_student_layers) - 1
        mapping.append(teacher_idx)

    return mapping