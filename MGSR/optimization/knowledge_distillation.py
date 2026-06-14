from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for child in value.values():
            tensor = _find_tensor(child)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for child in value:
            tensor = _find_tensor(child)
            if tensor is not None:
                return tensor
    return None


def _nested_get(outputs: Dict, *keys):
    value = outputs
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


class KnowledgeDistillationFramework(nn.Module):

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_matching_weight: float = 0.3,
    ):
        super().__init__()
        self.teacher_model = teacher_model.eval()
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.feature_matching_weight = feature_matching_weight
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        teacher_dim = getattr(teacher_model, "hidden_dim", None)
        student_dim = getattr(student_model, "hidden_dim", None)
        if teacher_dim and student_dim and teacher_dim != student_dim:
            self.user_adapter = nn.Linear(student_dim, teacher_dim)
            self.item_adapter = nn.Linear(student_dim, teacher_dim)
        else:
            self.user_adapter = nn.Identity()
            self.item_adapter = nn.Identity()
        self.relation_distillation = RelationDistillation(temperature)

    def forward(self, batch_data: Dict, return_teacher_outputs: bool = False):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch_data)
        student_outputs = self.student_model(**batch_data)
        losses = self._compute_distillation_losses(teacher_outputs, student_outputs)
        task_loss = student_outputs["total_loss"]
        total = (
            (1 - self.alpha) * task_loss
            + self.alpha * losses["knowledge_distillation_loss"]
            + self.feature_matching_weight * losses["feature_matching_loss"]
        )
        result = {
            "total_loss": total,
            "student_outputs": student_outputs,
            "distillation_losses": losses,
        }
        if return_teacher_outputs:
            result["teacher_outputs"] = teacher_outputs
        return result

    def _compute_distillation_losses(self, teacher_outputs: Dict, student_outputs: Dict):
        device_tensor = _find_tensor(student_outputs)
        if device_tensor is None:
            raise ValueError("Student outputs must contain tensors")
        zero = device_tensor.new_zeros(())

        teacher_logits = _nested_get(
            teacher_outputs, "retrieval_outputs", "coarse_results", "logits"
        )
        student_logits = _nested_get(
            student_outputs, "retrieval_outputs", "coarse_results", "logits"
        )
        kd_loss = zero
        if teacher_logits is not None and student_logits is not None:
            if teacher_logits.shape != student_logits.shape:
                raise ValueError("Teacher and student coarse logits must have matching shapes")
            teacher_probabilities = F.softmax(teacher_logits / self.temperature, dim=-1)
            student_log_probabilities = F.log_softmax(student_logits / self.temperature, dim=-1)
            kd_loss = (
                F.kl_div(student_log_probabilities, teacher_probabilities, reduction="batchmean")
                * self.temperature**2
            )

        feature_losses = []
        teacher_user = teacher_outputs.get("user_embeddings")
        student_user = student_outputs.get("user_embeddings")
        if teacher_user is not None and student_user is not None:
            feature_losses.append(F.mse_loss(self.user_adapter(student_user), teacher_user.detach()))
        teacher_item = teacher_outputs.get("item_embeddings")
        student_item = student_outputs.get("item_embeddings")
        if teacher_item is not None and student_item is not None:
            feature_losses.append(
                self.relation_distillation(
                    [teacher_item.detach()], [self.item_adapter(student_item)]
                )
            )
        feature_loss = torch.stack(feature_losses).mean() if feature_losses else zero
        return {
            "knowledge_distillation_loss": kd_loss,
            "feature_matching_loss": feature_loss,
        }


class AttentionTransfer(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, teacher_attentions: List[torch.Tensor], student_attentions: List[torch.Tensor]):
        losses = []
        for teacher, student in zip(teacher_attentions, student_attentions):
            if teacher.shape != student.shape:
                continue
            teacher_map = F.normalize(teacher.detach().reshape(teacher.size(0), -1), dim=-1)
            student_map = F.normalize(student.reshape(student.size(0), -1), dim=-1)
            losses.append(F.mse_loss(student_map, teacher_map))
        if not losses:
            reference = student_attentions[0] if student_attentions else teacher_attentions[0]
            return reference.new_zeros(())
        return self.beta * torch.stack(losses).mean()


class RelationDistillation(nn.Module):
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _relations(embeddings: torch.Tensor):
        normalized = F.normalize(embeddings.reshape(-1, embeddings.size(-1)), dim=-1)
        return normalized @ normalized.transpose(0, 1)

    def forward(self, teacher_embeddings: List[torch.Tensor], student_embeddings: List[torch.Tensor]):
        losses = []
        for teacher, student in zip(teacher_embeddings, student_embeddings):
            if teacher.size(-1) != student.size(-1):
                continue
            teacher_relations = self._relations(teacher.detach()) / self.temperature
            student_relations = self._relations(student) / self.temperature
            losses.append(F.mse_loss(student_relations, teacher_relations))
        if not losses:
            reference = student_embeddings[0] if student_embeddings else teacher_embeddings[0]
            return reference.new_zeros(())
        return torch.stack(losses).mean()


class ProgressiveKnowledgeDistillation(nn.Module):

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        stage_boundaries=(0, 5000, 10000),
        stage_alphas=(0.3, 0.5, 0.7),
        **kwargs,
    ):
        super().__init__()
        self.framework = KnowledgeDistillationFramework(teacher_model, student_model, **kwargs)
        self.stage_boundaries = stage_boundaries
        self.stage_alphas = stage_alphas

    def forward(self, batch_data: Dict, training_step: int = 0):
        stage = max(
            index for index, boundary in enumerate(self.stage_boundaries) if training_step >= boundary
        )
        self.framework.alpha = self.stage_alphas[min(stage, len(self.stage_alphas) - 1)]
        return self.framework(batch_data)


class OnlineKnowledgeDistillation(nn.Module):

    def __init__(self, teacher_models: Iterable[nn.Module], student_model: nn.Module, temperature: float = 4.0):
        super().__init__()
        self.teachers = nn.ModuleList(list(teacher_models))
        self.student = student_model
        self.temperature = temperature
        for teacher in self.teachers:
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad = False

    def forward(self, batch_data: Dict):
        student_outputs = self.student(**batch_data)
        student_logits = _nested_get(student_outputs, "retrieval_outputs", "coarse_results", "logits")
        if student_logits is None:
            raise ValueError("Student output must expose coarse retrieval logits")
        with torch.no_grad():
            teacher_logits = [
                _nested_get(teacher(**batch_data), "retrieval_outputs", "coarse_results", "logits")
                for teacher in self.teachers
            ]
            ensemble = torch.stack(teacher_logits).mean(dim=0)
        loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(ensemble / self.temperature, dim=-1),
            reduction="batchmean",
        ) * self.temperature**2
        return {"total_loss": student_outputs["total_loss"] + loss, "distillation_loss": loss}


class FeaturePyramidDistillation(nn.Module):

    def forward(self, teacher_features: List[torch.Tensor], student_features: List[torch.Tensor]):
        losses = [
            F.mse_loss(student, teacher.detach())
            for teacher, student in zip(teacher_features, student_features)
            if teacher.shape == student.shape
        ]
        if not losses:
            reference = student_features[0] if student_features else teacher_features[0]
            return reference.new_zeros(())
        return torch.stack(losses).mean()
