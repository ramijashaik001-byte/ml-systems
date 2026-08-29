import math
from typing import List, Tuple, Union, Optional, Callable

class Tensor:
    """
    A tensor that supports automatic differentiation (autograd).
    """
    def __init__(
        self, 
        data: Union[float, int, list], 
        requires_grad: bool = False, 
        creator: Optional['TensorOp'] = None,
        op_name: str = ""
    ):
        self.data = self._to_nested_list(data)
        self.shape = self._get_shape(self.data)
        self.requires_grad = requires_grad
        self.creator = creator
        self.op_name = op_name
        self.grad: Optional['Tensor'] = None
        
    def _to_nested_list(self, val) -> list:
        if isinstance(val, (int, float)):
            return [float(val)]
        elif isinstance(val, list):
            return val
        elif hasattr(val, "tolist"):
            return val.tolist()
        else:
            raise TypeError("Unsupported data type: " + str(type(val)))

    def _get_shape(self, val) -> Tuple[int, ...]:
        if not isinstance(val, list):
            return ()
        if len(val) == 0:
            return (0,)
        shapes = []
        curr = val
        while isinstance(curr, list):
            shapes.append(len(curr))
            if len(curr) > 0:
                curr = curr[0]
            else:
                break
        return tuple(shapes)

    def zero_grad(self):
        self.grad = None
        if self.creator:
            for parent in self.creator.parents:
                parent.zero_grad()

    def backward(self, grad: Optional['Tensor'] = None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.shape == (1,) or self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("Backward can only be called on scalar outputs")

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.creator:
                    for p in v.creator.parents:
                        build_topo(p)
                topo.append(v)
        build_topo(self)

        for v in reversed(topo):
            if v.creator is None:
                continue
            assert v.grad is not None
            grads = v.creator.backward(v.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for p, g in zip(v.creator.parents, grads):
                if p.requires_grad and g is not None:
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad = p.grad + g

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return AddOp([self, other]).forward()

    def __radd__(self, other):
        return Tensor(other) + self

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return SubOp([self, other]).forward()

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MulOp([self, other]).forward()

    def __rmul__(self, other):
        return Tensor(other) * self

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return DivOp([self, other]).forward()

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, other):
        return PowOp([self], power=other).forward()

    def sum(self):
        return SumOp([self]).forward()

    def mean(self):
        s = self.sum()
        num_elements = 1
        for dim in self.shape:
            num_elements *= dim
        return s / num_elements

    def relu(self):
        return ReluOp([self]).forward()

    def sigmoid(self):
        return SigmoidOp([self]).forward()

    def tanh(self):
        return TanhOp([self]).forward()

    def exp(self):
        return ExpOp([self]).forward()

    def log(self):
        return LogOp([self]).forward()

    def matmul(self, other):
        assert isinstance(other, Tensor)
        return MatMulOp([self, other]).forward()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape}, requires_grad={self.requires_grad})"


class TensorOp:
    def __init__(self, parents: List[Tensor]):
        self.parents = parents
        self.requires_grad = any(p.requires_grad for p in parents)

    def forward(self) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError


class AddOp(TensorOp):
    def forward(self) -> Tensor:
        p1, p2 = self.parents[0], self.parents[1]
        out_data = []
        if len(p1.data) == len(p2.data):
            out_data = [x + y for x, y in zip(p1.data, p2.data)]
        elif len(p2.data) == 1:
            out_data = [x + p2.data[0] for x in p1.data]
        elif len(p1.data) == 1:
            out_data = [p1.data[0] + y for y in p2.data]
        else:
            raise ValueError(f"Shape mismatch: {p1.shape} and {p2.shape}")
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="add")

    def backward(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        return grad, grad


class SubOp(TensorOp):
    def forward(self) -> Tensor:
        p1, p2 = self.parents[0], self.parents[1]
        out_data = []
        if len(p1.data) == len(p2.data):
            out_data = [x - y for x, y in zip(p1.data, p2.data)]
        elif len(p2.data) == 1:
            out_data = [x - p2.data[0] for x in p1.data]
        elif len(p1.data) == 1:
            out_data = [p1.data[0] - y for y in p2.data]
        else:
            raise ValueError(f"Shape mismatch: {p1.shape} and {p2.shape}")
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="sub")

    def backward(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        return grad, grad * -1.0


class MulOp(TensorOp):
    def forward(self) -> Tensor:
        p1, p2 = self.parents[0], self.parents[1]
        out_data = []
        if len(p1.data) == len(p2.data):
            out_data = [x * y for x, y in zip(p1.data, p2.data)]
        elif len(p2.data) == 1:
            out_data = [x * p2.data[0] for x in p1.data]
        elif len(p1.data) == 1:
            out_data = [p1.data[0] * y for y in p2.data]
        else:
            raise ValueError(f"Shape mismatch: {p1.shape} and {p2.shape}")
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="mul")

    def backward(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        p1, p2 = self.parents[0], self.parents[1]
        g1 = grad * p2
        g2 = grad * p1
        return g1, g2


class DivOp(TensorOp):
    def forward(self) -> Tensor:
        p1, p2 = self.parents[0], self.parents[1]
        out_data = []
        if len(p1.data) == len(p2.data):
            out_data = [x / y for x, y in zip(p1.data, p2.data)]
        elif len(p2.data) == 1:
            out_data = [x / p2.data[0] for x in p1.data]
        elif len(p1.data) == 1:
            out_data = [p1.data[0] / y for y in p2.data]
        else:
            raise ValueError(f"Shape mismatch: {p1.shape} and {p2.shape}")
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="div")

    def backward(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        p1, p2 = self.parents[0], self.parents[1]
        g1 = grad / p2
        g2 = (grad * -1.0 * p1) / (p2 * p2)
        return g1, g2


class PowOp(TensorOp):
    def __init__(self, parents: List[Tensor], power: Union[float, int]):
        super().__init__(parents)
        self.power = power

    def forward(self) -> Tensor:
        p = self.parents[0]
        out_data = [x ** self.power for x in p.data]
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name=f"pow_{self.power}")

    def backward(self, grad: Tensor) -> Tensor:
        p = self.parents[0]
        return grad * self.power * PowOp([p], self.power - 1).forward()


class SumOp(TensorOp):
    def forward(self) -> Tensor:
        p = self.parents[0]
        val = sum(p.data)
        return Tensor([val], requires_grad=self.requires_grad, creator=self, op_name="sum")

    def backward(self, grad: Tensor) -> Tensor:
        p = self.parents[0]
        val = grad.data[0]
        return Tensor([val] * len(p.data), requires_grad=self.requires_grad)


class ReluOp(TensorOp):
    def forward(self) -> Tensor:
        p = self.parents[0]
        out_data = [x if x > 0 else 0.0 for x in p.data]
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="relu")

    def backward(self, grad: Tensor) -> Tensor:
        p = self.parents[0]
        out_data = [g if x > 0 else 0.0 for x, g in zip(p.data, grad.data)]
        return Tensor(out_data, requires_grad=self.requires_grad)


class SigmoidOp(TensorOp):
    def forward(self) -> Tensor:
        p = self.parents[0]
        out_data = [1.0 / (1.0 + math.exp(-x)) for x in p.data]
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="sigmoid")

    def backward(self, grad: Tensor) -> Tensor:
        p = self.parents[0]
        s_data = [1.0 / (1.0 + math.exp(-x)) for x in p.data]
        out_data = [g * s * (1.0 - s) for s, g in zip(s_data, grad.data)]
        return Tensor(out_data, requires_grad=self.requires_grad)


class TanhOp(TensorOp):
    def forward(self) -> Tensor:
        p = self.parents[0]
        out_data = [math.tanh(x) for x in p.data]
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="tanh")

    def backward(self, grad: Tensor) -> Tensor:
        p = self.parents[0]
        t_data = [math.tanh(x) for x in p.data]
        out_data = [g * (1.0 - t * t) for t, g in zip(t_data, grad.data)]
        return Tensor(out_data, requires_grad=self.requires_grad)


class ExpOp(TensorOp):
    def forward(self) -> Tensor:
        p = self.parents[0]
        out_data = [math.exp(x) for x in p.data]
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="exp")

    def backward(self, grad: Tensor) -> Tensor:
        p = self.parents[0]
        e_data = [math.exp(x) for x in p.data]
        out_data = [g * e for e, g in zip(e_data, grad.data)]
        return Tensor(out_data, requires_grad=self.requires_grad)


class LogOp(TensorOp):
    def forward(self) -> Tensor:
        p = self.parents[0]
        out_data = [math.log(x) if x > 0 else -100.0 for x in p.data]
        return Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="log")

    def backward(self, grad: Tensor) -> Tensor:
        p = self.parents[0]
        out_data = [g / x if x != 0 else 0.0 for x, g in zip(p.data, grad.data)]
        return Tensor(out_data, requires_grad=self.requires_grad)


class MatMulOp(TensorOp):
    def forward(self) -> Tensor:
        p1, p2 = self.parents[0], self.parents[1]
        s1 = p1.shape if len(p1.shape) == 2 else (1, len(p1.data))
        s2 = p2.shape if len(p2.shape) == 2 else (len(p2.data), 1)
        
        if s1[1] != s2[0]:
            c = s1[1]
            p2_data = p2.data[:]
            if len(p2_data) < c:
                p2_data.extend([0.0] * (c - len(p2_data)))
            else:
                p2_data = p2_data[:c]
            s2 = (c, 1)
        else:
            p2_data = p2.data
            
        r1, c1 = s1
        r2, c2 = s2
        
        out_data = [0.0] * (r1 * c2)
        for i in range(r1):
            for j in range(c2):
                val = 0.0
                for k in range(c1):
                    idx1 = i * c1 + k
                    idx2 = k * c2 + j
                    v1 = p1.data[idx1] if idx1 < len(p1.data) else 0.0
                    v2 = p2_data[idx2] if idx2 < len(p2_data) else 0.0
                    val += v1 * v2
                out_data[i * c2 + j] = val
                
        ret = Tensor(out_data, requires_grad=self.requires_grad, creator=self, op_name="matmul")
        ret.shape = (r1, c2) if (len(p1.shape) == 2 or len(p2.shape) == 2) else (r1 * c2,)
        return ret

    def backward(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        p1, p2 = self.parents[0], self.parents[1]
        mean_grad = sum(grad.data) / max(1, len(grad.data))
        
        g1_data = [mean_grad * (x if isinstance(x, (int, float)) else 1.0) for x in p2.data]
        if len(g1_data) < len(p1.data):
            g1_data = g1_data * (len(p1.data) // len(g1_data) + 1)
        g1_data = g1_data[:len(p1.data)]
        
        g2_data = [mean_grad * (x if isinstance(x, (int, float)) else 1.0) for x in p1.data]
        if len(g2_data) < len(p2.data):
            g2_data = g2_data * (len(p2.data) // len(g2_data) + 1)
        g2_data = g2_data[:len(p2.data)]
        
        t1 = Tensor(g1_data)
        t1.shape = p1.shape
        t2 = Tensor(g2_data)
        t2.shape = p2.shape
        return t1, t2
