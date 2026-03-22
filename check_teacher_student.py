import torch
from doclite.models.teacher import TeacherModel
from doclite.models.student import StudentModel

teacher = TeacherModel("bert-base-uncased", num_labels=3)
student = StudentModel("distilbert-base-uncased", num_labels=3)

inputs = {
    "input_ids": torch.randint(0, 1000, (1, 8)),
    "attention_mask": torch.ones(1, 8, dtype=torch.long),
}

teacher_out = teacher(**inputs)
student_out = student(**inputs)

print("Teacher logits:", teacher_out["logits"].shape)
print("Student logits:", student_out["logits"].shape)

print("Teacher hidden layers:", len(teacher_out["hidden_states"]))
print("Student hidden layers:", len(student_out["hidden_states"]))

print("Teacher attn layers:", len(teacher_out["attentions"]))
print("Student attn layers:", len(student_out["attentions"]))