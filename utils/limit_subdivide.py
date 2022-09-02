"""
    Loop Limit Subdvide helper class
"""
import torch
import loop_limitation

class limitation_evaluate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, loop_obj):
        limitation = loop_obj.compute_limitation(input)
        jacobian = loop_obj.get_J()
        ctx.in1 = jacobian
        return limitation

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.in1.T
        out = torch.matmul(grad,grad_output)      
        return out, None


class LimitSubdivide():
    def __init__(self, vertices, faces) -> None:
        self.loop_limit = loop_limitation.loop_limitation()
        self.loop_limit.init_J(vertices.to('cpu').double(), faces.to('cpu').int())
        self.compute_limit = limitation_evaluate.apply

    def get_limit(self, vertices):
        new_verts  = self.compute_limit(vertices.to('cpu').double(), self.loop_limit)
        return new_verts