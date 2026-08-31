import unittest
from nexusml.core.tensor import Tensor
from nexusml.core.ops import conv2d, maxpool2d, dropout, batchnorm2d

class TestTensorOps(unittest.TestCase):

    def test_tensor_ops_case_1(self):
        """Test Case 1 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_2(self):
        """Test Case 2 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_3(self):
        """Test Case 3 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_4(self):
        """Test Case 4 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_5(self):
        """Test Case 5 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_6(self):
        """Test Case 6 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_7(self):
        """Test Case 7 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_8(self):
        """Test Case 8 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_9(self):
        """Test Case 9 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_10(self):
        """Test Case 10 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_11(self):
        """Test Case 11 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_12(self):
        """Test Case 12 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_13(self):
        """Test Case 13 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_14(self):
        """Test Case 14 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_15(self):
        """Test Case 15 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_16(self):
        """Test Case 16 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_17(self):
        """Test Case 17 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_18(self):
        """Test Case 18 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_19(self):
        """Test Case 19 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_20(self):
        """Test Case 20 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_21(self):
        """Test Case 21 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_22(self):
        """Test Case 22 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_23(self):
        """Test Case 23 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_24(self):
        """Test Case 24 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_25(self):
        """Test Case 25 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_26(self):
        """Test Case 26 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_27(self):
        """Test Case 27 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_28(self):
        """Test Case 28 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_29(self):
        """Test Case 29 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_30(self):
        """Test Case 30 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_31(self):
        """Test Case 31 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_32(self):
        """Test Case 32 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_33(self):
        """Test Case 33 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_34(self):
        """Test Case 34 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_35(self):
        """Test Case 35 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_36(self):
        """Test Case 36 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_37(self):
        """Test Case 37 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_38(self):
        """Test Case 38 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_39(self):
        """Test Case 39 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_40(self):
        """Test Case 40 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_41(self):
        """Test Case 41 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_42(self):
        """Test Case 42 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_43(self):
        """Test Case 43 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_44(self):
        """Test Case 44 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_45(self):
        """Test Case 45 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_46(self):
        """Test Case 46 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_47(self):
        """Test Case 47 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_48(self):
        """Test Case 48 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_49(self):
        """Test Case 49 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_50(self):
        """Test Case 50 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_51(self):
        """Test Case 51 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_52(self):
        """Test Case 52 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_53(self):
        """Test Case 53 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_54(self):
        """Test Case 54 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_55(self):
        """Test Case 55 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_56(self):
        """Test Case 56 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_57(self):
        """Test Case 57 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_58(self):
        """Test Case 58 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_59(self):
        """Test Case 59 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_60(self):
        """Test Case 60 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_61(self):
        """Test Case 61 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_62(self):
        """Test Case 62 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_63(self):
        """Test Case 63 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_64(self):
        """Test Case 64 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_65(self):
        """Test Case 65 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_66(self):
        """Test Case 66 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_67(self):
        """Test Case 67 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_68(self):
        """Test Case 68 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_69(self):
        """Test Case 69 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_70(self):
        """Test Case 70 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_71(self):
        """Test Case 71 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_72(self):
        """Test Case 72 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_73(self):
        """Test Case 73 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_74(self):
        """Test Case 74 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_75(self):
        """Test Case 75 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_76(self):
        """Test Case 76 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_77(self):
        """Test Case 77 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_78(self):
        """Test Case 78 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_79(self):
        """Test Case 79 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_80(self):
        """Test Case 80 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_81(self):
        """Test Case 81 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_82(self):
        """Test Case 82 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_83(self):
        """Test Case 83 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_84(self):
        """Test Case 84 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_85(self):
        """Test Case 85 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_86(self):
        """Test Case 86 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_87(self):
        """Test Case 87 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_88(self):
        """Test Case 88 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_89(self):
        """Test Case 89 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_90(self):
        """Test Case 90 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_91(self):
        """Test Case 91 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_92(self):
        """Test Case 92 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_93(self):
        """Test Case 93 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_94(self):
        """Test Case 94 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_95(self):
        """Test Case 95 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_96(self):
        """Test Case 96 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_97(self):
        """Test Case 97 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_98(self):
        """Test Case 98 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_99(self):
        """Test Case 99 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_100(self):
        """Test Case 100 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_101(self):
        """Test Case 101 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_102(self):
        """Test Case 102 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_103(self):
        """Test Case 103 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_104(self):
        """Test Case 104 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_105(self):
        """Test Case 105 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_106(self):
        """Test Case 106 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_107(self):
        """Test Case 107 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_108(self):
        """Test Case 108 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_109(self):
        """Test Case 109 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_110(self):
        """Test Case 110 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_111(self):
        """Test Case 111 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_112(self):
        """Test Case 112 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_113(self):
        """Test Case 113 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_114(self):
        """Test Case 114 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_115(self):
        """Test Case 115 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_116(self):
        """Test Case 116 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_117(self):
        """Test Case 117 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_118(self):
        """Test Case 118 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_119(self):
        """Test Case 119 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_120(self):
        """Test Case 120 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_121(self):
        """Test Case 121 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_122(self):
        """Test Case 122 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_123(self):
        """Test Case 123 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_124(self):
        """Test Case 124 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_125(self):
        """Test Case 125 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_126(self):
        """Test Case 126 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_127(self):
        """Test Case 127 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_128(self):
        """Test Case 128 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_129(self):
        """Test Case 129 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_130(self):
        """Test Case 130 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_131(self):
        """Test Case 131 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_132(self):
        """Test Case 132 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_133(self):
        """Test Case 133 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_134(self):
        """Test Case 134 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_135(self):
        """Test Case 135 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_136(self):
        """Test Case 136 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_137(self):
        """Test Case 137 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_138(self):
        """Test Case 138 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_139(self):
        """Test Case 139 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_140(self):
        """Test Case 140 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_141(self):
        """Test Case 141 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_142(self):
        """Test Case 142 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_143(self):
        """Test Case 143 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_144(self):
        """Test Case 144 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_145(self):
        """Test Case 145 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_146(self):
        """Test Case 146 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_147(self):
        """Test Case 147 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_148(self):
        """Test Case 148 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_149(self):
        """Test Case 149 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_150(self):
        """Test Case 150 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_151(self):
        """Test Case 151 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_152(self):
        """Test Case 152 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_153(self):
        """Test Case 153 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_154(self):
        """Test Case 154 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_155(self):
        """Test Case 155 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_156(self):
        """Test Case 156 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_157(self):
        """Test Case 157 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_158(self):
        """Test Case 158 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_159(self):
        """Test Case 159 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_160(self):
        """Test Case 160 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_161(self):
        """Test Case 161 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_162(self):
        """Test Case 162 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_163(self):
        """Test Case 163 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_164(self):
        """Test Case 164 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_165(self):
        """Test Case 165 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_166(self):
        """Test Case 166 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_167(self):
        """Test Case 167 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_168(self):
        """Test Case 168 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_169(self):
        """Test Case 169 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_170(self):
        """Test Case 170 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_171(self):
        """Test Case 171 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_172(self):
        """Test Case 172 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_173(self):
        """Test Case 173 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_174(self):
        """Test Case 174 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_175(self):
        """Test Case 175 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_176(self):
        """Test Case 176 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_177(self):
        """Test Case 177 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_178(self):
        """Test Case 178 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_179(self):
        """Test Case 179 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_180(self):
        """Test Case 180 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_181(self):
        """Test Case 181 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_182(self):
        """Test Case 182 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_183(self):
        """Test Case 183 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_184(self):
        """Test Case 184 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_185(self):
        """Test Case 185 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_186(self):
        """Test Case 186 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_187(self):
        """Test Case 187 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_188(self):
        """Test Case 188 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_189(self):
        """Test Case 189 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_190(self):
        """Test Case 190 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_191(self):
        """Test Case 191 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_192(self):
        """Test Case 192 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_193(self):
        """Test Case 193 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_194(self):
        """Test Case 194 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_195(self):
        """Test Case 195 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_196(self):
        """Test Case 196 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_197(self):
        """Test Case 197 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_198(self):
        """Test Case 198 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_199(self):
        """Test Case 199 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_200(self):
        """Test Case 200 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_201(self):
        """Test Case 201 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_202(self):
        """Test Case 202 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_203(self):
        """Test Case 203 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_204(self):
        """Test Case 204 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_205(self):
        """Test Case 205 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_206(self):
        """Test Case 206 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_207(self):
        """Test Case 207 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_208(self):
        """Test Case 208 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_209(self):
        """Test Case 209 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_210(self):
        """Test Case 210 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_211(self):
        """Test Case 211 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_212(self):
        """Test Case 212 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_213(self):
        """Test Case 213 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_214(self):
        """Test Case 214 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_215(self):
        """Test Case 215 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_216(self):
        """Test Case 216 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_217(self):
        """Test Case 217 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_218(self):
        """Test Case 218 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_219(self):
        """Test Case 219 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_220(self):
        """Test Case 220 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_221(self):
        """Test Case 221 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_222(self):
        """Test Case 222 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_223(self):
        """Test Case 223 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_224(self):
        """Test Case 224 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_225(self):
        """Test Case 225 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_226(self):
        """Test Case 226 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_227(self):
        """Test Case 227 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_228(self):
        """Test Case 228 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_229(self):
        """Test Case 229 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_230(self):
        """Test Case 230 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_231(self):
        """Test Case 231 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_232(self):
        """Test Case 232 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_233(self):
        """Test Case 233 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_234(self):
        """Test Case 234 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_235(self):
        """Test Case 235 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_236(self):
        """Test Case 236 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_237(self):
        """Test Case 237 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_238(self):
        """Test Case 238 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_239(self):
        """Test Case 239 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_240(self):
        """Test Case 240 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_241(self):
        """Test Case 241 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_242(self):
        """Test Case 242 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_243(self):
        """Test Case 243 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_244(self):
        """Test Case 244 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_245(self):
        """Test Case 245 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_246(self):
        """Test Case 246 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_247(self):
        """Test Case 247 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_248(self):
        """Test Case 248 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_249(self):
        """Test Case 249 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_250(self):
        """Test Case 250 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_251(self):
        """Test Case 251 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_252(self):
        """Test Case 252 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_253(self):
        """Test Case 253 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_254(self):
        """Test Case 254 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_255(self):
        """Test Case 255 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_256(self):
        """Test Case 256 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_257(self):
        """Test Case 257 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_258(self):
        """Test Case 258 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_259(self):
        """Test Case 259 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_260(self):
        """Test Case 260 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_261(self):
        """Test Case 261 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_262(self):
        """Test Case 262 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_263(self):
        """Test Case 263 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_264(self):
        """Test Case 264 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_265(self):
        """Test Case 265 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_266(self):
        """Test Case 266 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_267(self):
        """Test Case 267 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_268(self):
        """Test Case 268 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_269(self):
        """Test Case 269 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_270(self):
        """Test Case 270 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_271(self):
        """Test Case 271 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_272(self):
        """Test Case 272 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_273(self):
        """Test Case 273 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_274(self):
        """Test Case 274 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_275(self):
        """Test Case 275 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_276(self):
        """Test Case 276 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_277(self):
        """Test Case 277 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_278(self):
        """Test Case 278 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_279(self):
        """Test Case 279 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_280(self):
        """Test Case 280 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_281(self):
        """Test Case 281 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_282(self):
        """Test Case 282 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_283(self):
        """Test Case 283 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_284(self):
        """Test Case 284 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_285(self):
        """Test Case 285 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_286(self):
        """Test Case 286 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_287(self):
        """Test Case 287 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_288(self):
        """Test Case 288 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_289(self):
        """Test Case 289 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_290(self):
        """Test Case 290 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_291(self):
        """Test Case 291 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_292(self):
        """Test Case 292 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_293(self):
        """Test Case 293 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_294(self):
        """Test Case 294 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_295(self):
        """Test Case 295 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_296(self):
        """Test Case 296 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_297(self):
        """Test Case 297 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_298(self):
        """Test Case 298 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_299(self):
        """Test Case 299 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_300(self):
        """Test Case 300 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_301(self):
        """Test Case 301 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_302(self):
        """Test Case 302 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_303(self):
        """Test Case 303 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_304(self):
        """Test Case 304 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_305(self):
        """Test Case 305 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_306(self):
        """Test Case 306 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_307(self):
        """Test Case 307 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_308(self):
        """Test Case 308 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_309(self):
        """Test Case 309 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_310(self):
        """Test Case 310 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_311(self):
        """Test Case 311 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_312(self):
        """Test Case 312 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_313(self):
        """Test Case 313 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_314(self):
        """Test Case 314 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_315(self):
        """Test Case 315 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_316(self):
        """Test Case 316 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_317(self):
        """Test Case 317 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_318(self):
        """Test Case 318 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_319(self):
        """Test Case 319 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_320(self):
        """Test Case 320 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_321(self):
        """Test Case 321 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_322(self):
        """Test Case 322 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_323(self):
        """Test Case 323 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_324(self):
        """Test Case 324 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_325(self):
        """Test Case 325 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_326(self):
        """Test Case 326 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_327(self):
        """Test Case 327 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_328(self):
        """Test Case 328 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_329(self):
        """Test Case 329 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_330(self):
        """Test Case 330 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_331(self):
        """Test Case 331 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_332(self):
        """Test Case 332 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_333(self):
        """Test Case 333 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_334(self):
        """Test Case 334 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_335(self):
        """Test Case 335 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_336(self):
        """Test Case 336 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_337(self):
        """Test Case 337 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_338(self):
        """Test Case 338 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_339(self):
        """Test Case 339 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_340(self):
        """Test Case 340 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_341(self):
        """Test Case 341 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_342(self):
        """Test Case 342 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_343(self):
        """Test Case 343 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_344(self):
        """Test Case 344 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_345(self):
        """Test Case 345 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_346(self):
        """Test Case 346 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_347(self):
        """Test Case 347 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_348(self):
        """Test Case 348 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_349(self):
        """Test Case 349 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_350(self):
        """Test Case 350 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_351(self):
        """Test Case 351 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_352(self):
        """Test Case 352 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_353(self):
        """Test Case 353 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_354(self):
        """Test Case 354 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_355(self):
        """Test Case 355 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_356(self):
        """Test Case 356 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_357(self):
        """Test Case 357 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_358(self):
        """Test Case 358 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_359(self):
        """Test Case 359 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_360(self):
        """Test Case 360 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_361(self):
        """Test Case 361 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_362(self):
        """Test Case 362 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_363(self):
        """Test Case 363 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_364(self):
        """Test Case 364 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_365(self):
        """Test Case 365 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_366(self):
        """Test Case 366 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_367(self):
        """Test Case 367 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_368(self):
        """Test Case 368 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_369(self):
        """Test Case 369 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_370(self):
        """Test Case 370 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_371(self):
        """Test Case 371 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_372(self):
        """Test Case 372 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_373(self):
        """Test Case 373 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_374(self):
        """Test Case 374 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_375(self):
        """Test Case 375 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_376(self):
        """Test Case 376 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_377(self):
        """Test Case 377 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_378(self):
        """Test Case 378 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_379(self):
        """Test Case 379 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_380(self):
        """Test Case 380 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_381(self):
        """Test Case 381 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_382(self):
        """Test Case 382 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_383(self):
        """Test Case 383 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_384(self):
        """Test Case 384 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_385(self):
        """Test Case 385 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_386(self):
        """Test Case 386 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_387(self):
        """Test Case 387 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_388(self):
        """Test Case 388 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_389(self):
        """Test Case 389 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_390(self):
        """Test Case 390 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_391(self):
        """Test Case 391 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_392(self):
        """Test Case 392 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_393(self):
        """Test Case 393 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_394(self):
        """Test Case 394 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_395(self):
        """Test Case 395 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_396(self):
        """Test Case 396 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_397(self):
        """Test Case 397 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_398(self):
        """Test Case 398 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_399(self):
        """Test Case 399 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_400(self):
        """Test Case 400 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_401(self):
        """Test Case 401 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_402(self):
        """Test Case 402 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_403(self):
        """Test Case 403 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_404(self):
        """Test Case 404 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_405(self):
        """Test Case 405 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_406(self):
        """Test Case 406 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_407(self):
        """Test Case 407 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_408(self):
        """Test Case 408 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_409(self):
        """Test Case 409 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_410(self):
        """Test Case 410 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_411(self):
        """Test Case 411 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_412(self):
        """Test Case 412 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_413(self):
        """Test Case 413 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_414(self):
        """Test Case 414 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_415(self):
        """Test Case 415 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_416(self):
        """Test Case 416 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_417(self):
        """Test Case 417 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_418(self):
        """Test Case 418 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_419(self):
        """Test Case 419 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_420(self):
        """Test Case 420 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_421(self):
        """Test Case 421 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_422(self):
        """Test Case 422 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_423(self):
        """Test Case 423 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_424(self):
        """Test Case 424 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_425(self):
        """Test Case 425 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_426(self):
        """Test Case 426 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_427(self):
        """Test Case 427 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_428(self):
        """Test Case 428 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_429(self):
        """Test Case 429 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_430(self):
        """Test Case 430 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_431(self):
        """Test Case 431 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_432(self):
        """Test Case 432 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_433(self):
        """Test Case 433 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_434(self):
        """Test Case 434 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_435(self):
        """Test Case 435 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_436(self):
        """Test Case 436 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_437(self):
        """Test Case 437 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_438(self):
        """Test Case 438 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_439(self):
        """Test Case 439 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_440(self):
        """Test Case 440 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_441(self):
        """Test Case 441 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_442(self):
        """Test Case 442 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_443(self):
        """Test Case 443 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_444(self):
        """Test Case 444 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_445(self):
        """Test Case 445 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_446(self):
        """Test Case 446 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_447(self):
        """Test Case 447 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_448(self):
        """Test Case 448 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_449(self):
        """Test Case 449 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_450(self):
        """Test Case 450 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_451(self):
        """Test Case 451 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_452(self):
        """Test Case 452 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_453(self):
        """Test Case 453 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_454(self):
        """Test Case 454 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_455(self):
        """Test Case 455 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_456(self):
        """Test Case 456 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_457(self):
        """Test Case 457 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_458(self):
        """Test Case 458 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_459(self):
        """Test Case 459 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_460(self):
        """Test Case 460 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_461(self):
        """Test Case 461 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_462(self):
        """Test Case 462 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_463(self):
        """Test Case 463 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_464(self):
        """Test Case 464 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_465(self):
        """Test Case 465 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_466(self):
        """Test Case 466 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_467(self):
        """Test Case 467 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_468(self):
        """Test Case 468 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_469(self):
        """Test Case 469 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_470(self):
        """Test Case 470 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_471(self):
        """Test Case 471 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_472(self):
        """Test Case 472 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_473(self):
        """Test Case 473 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_474(self):
        """Test Case 474 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_475(self):
        """Test Case 475 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_476(self):
        """Test Case 476 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_477(self):
        """Test Case 477 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_478(self):
        """Test Case 478 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_479(self):
        """Test Case 479 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_480(self):
        """Test Case 480 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_481(self):
        """Test Case 481 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_482(self):
        """Test Case 482 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_483(self):
        """Test Case 483 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_484(self):
        """Test Case 484 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_485(self):
        """Test Case 485 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_486(self):
        """Test Case 486 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_487(self):
        """Test Case 487 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_488(self):
        """Test Case 488 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_489(self):
        """Test Case 489 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_490(self):
        """Test Case 490 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_491(self):
        """Test Case 491 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_492(self):
        """Test Case 492 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_493(self):
        """Test Case 493 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_494(self):
        """Test Case 494 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_495(self):
        """Test Case 495 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_496(self):
        """Test Case 496 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_497(self):
        """Test Case 497 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_498(self):
        """Test Case 498 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_499(self):
        """Test Case 499 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_500(self):
        """Test Case 500 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_501(self):
        """Test Case 501 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_502(self):
        """Test Case 502 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_503(self):
        """Test Case 503 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_504(self):
        """Test Case 504 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_505(self):
        """Test Case 505 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_506(self):
        """Test Case 506 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_507(self):
        """Test Case 507 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_508(self):
        """Test Case 508 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_509(self):
        """Test Case 509 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_510(self):
        """Test Case 510 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_511(self):
        """Test Case 511 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_512(self):
        """Test Case 512 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_513(self):
        """Test Case 513 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_514(self):
        """Test Case 514 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_515(self):
        """Test Case 515 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_516(self):
        """Test Case 516 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_517(self):
        """Test Case 517 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_518(self):
        """Test Case 518 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_519(self):
        """Test Case 519 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_520(self):
        """Test Case 520 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_521(self):
        """Test Case 521 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_522(self):
        """Test Case 522 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_523(self):
        """Test Case 523 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_524(self):
        """Test Case 524 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_525(self):
        """Test Case 525 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_526(self):
        """Test Case 526 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_527(self):
        """Test Case 527 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_528(self):
        """Test Case 528 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_529(self):
        """Test Case 529 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_530(self):
        """Test Case 530 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_531(self):
        """Test Case 531 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_532(self):
        """Test Case 532 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_533(self):
        """Test Case 533 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_534(self):
        """Test Case 534 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_535(self):
        """Test Case 535 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_536(self):
        """Test Case 536 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_537(self):
        """Test Case 537 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_538(self):
        """Test Case 538 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_539(self):
        """Test Case 539 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_540(self):
        """Test Case 540 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_541(self):
        """Test Case 541 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_542(self):
        """Test Case 542 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_543(self):
        """Test Case 543 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_544(self):
        """Test Case 544 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_545(self):
        """Test Case 545 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_546(self):
        """Test Case 546 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_547(self):
        """Test Case 547 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_548(self):
        """Test Case 548 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_549(self):
        """Test Case 549 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_550(self):
        """Test Case 550 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_551(self):
        """Test Case 551 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_552(self):
        """Test Case 552 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_553(self):
        """Test Case 553 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_554(self):
        """Test Case 554 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_555(self):
        """Test Case 555 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_556(self):
        """Test Case 556 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_557(self):
        """Test Case 557 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_558(self):
        """Test Case 558 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_559(self):
        """Test Case 559 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_560(self):
        """Test Case 560 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_561(self):
        """Test Case 561 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_562(self):
        """Test Case 562 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_563(self):
        """Test Case 563 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_564(self):
        """Test Case 564 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_565(self):
        """Test Case 565 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_566(self):
        """Test Case 566 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_567(self):
        """Test Case 567 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_568(self):
        """Test Case 568 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_569(self):
        """Test Case 569 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_570(self):
        """Test Case 570 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_571(self):
        """Test Case 571 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_572(self):
        """Test Case 572 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_573(self):
        """Test Case 573 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_574(self):
        """Test Case 574 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_575(self):
        """Test Case 575 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_576(self):
        """Test Case 576 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_577(self):
        """Test Case 577 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_578(self):
        """Test Case 578 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_579(self):
        """Test Case 579 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_580(self):
        """Test Case 580 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_581(self):
        """Test Case 581 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_582(self):
        """Test Case 582 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_583(self):
        """Test Case 583 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_584(self):
        """Test Case 584 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_585(self):
        """Test Case 585 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_586(self):
        """Test Case 586 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_587(self):
        """Test Case 587 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_588(self):
        """Test Case 588 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_589(self):
        """Test Case 589 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_590(self):
        """Test Case 590 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_591(self):
        """Test Case 591 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_592(self):
        """Test Case 592 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_593(self):
        """Test Case 593 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_594(self):
        """Test Case 594 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_595(self):
        """Test Case 595 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_596(self):
        """Test Case 596 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_597(self):
        """Test Case 597 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_598(self):
        """Test Case 598 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_599(self):
        """Test Case 599 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_600(self):
        """Test Case 600 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_601(self):
        """Test Case 601 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_602(self):
        """Test Case 602 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_603(self):
        """Test Case 603 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_604(self):
        """Test Case 604 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_605(self):
        """Test Case 605 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_606(self):
        """Test Case 606 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_607(self):
        """Test Case 607 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_608(self):
        """Test Case 608 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_609(self):
        """Test Case 609 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_610(self):
        """Test Case 610 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_611(self):
        """Test Case 611 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_612(self):
        """Test Case 612 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_613(self):
        """Test Case 613 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_614(self):
        """Test Case 614 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_615(self):
        """Test Case 615 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_616(self):
        """Test Case 616 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_617(self):
        """Test Case 617 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_618(self):
        """Test Case 618 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_619(self):
        """Test Case 619 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_620(self):
        """Test Case 620 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_621(self):
        """Test Case 621 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_622(self):
        """Test Case 622 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_623(self):
        """Test Case 623 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_624(self):
        """Test Case 624 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_625(self):
        """Test Case 625 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_626(self):
        """Test Case 626 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_627(self):
        """Test Case 627 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_628(self):
        """Test Case 628 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_629(self):
        """Test Case 629 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_630(self):
        """Test Case 630 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_631(self):
        """Test Case 631 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_632(self):
        """Test Case 632 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_633(self):
        """Test Case 633 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_634(self):
        """Test Case 634 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_635(self):
        """Test Case 635 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_636(self):
        """Test Case 636 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_637(self):
        """Test Case 637 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_638(self):
        """Test Case 638 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_639(self):
        """Test Case 639 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_640(self):
        """Test Case 640 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_641(self):
        """Test Case 641 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_642(self):
        """Test Case 642 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_643(self):
        """Test Case 643 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_644(self):
        """Test Case 644 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_645(self):
        """Test Case 645 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_646(self):
        """Test Case 646 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_647(self):
        """Test Case 647 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_648(self):
        """Test Case 648 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_649(self):
        """Test Case 649 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_650(self):
        """Test Case 650 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_651(self):
        """Test Case 651 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_652(self):
        """Test Case 652 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_653(self):
        """Test Case 653 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_654(self):
        """Test Case 654 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_655(self):
        """Test Case 655 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_656(self):
        """Test Case 656 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_657(self):
        """Test Case 657 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_658(self):
        """Test Case 658 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_659(self):
        """Test Case 659 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_660(self):
        """Test Case 660 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_661(self):
        """Test Case 661 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_662(self):
        """Test Case 662 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_663(self):
        """Test Case 663 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_664(self):
        """Test Case 664 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_665(self):
        """Test Case 665 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_666(self):
        """Test Case 666 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_667(self):
        """Test Case 667 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_668(self):
        """Test Case 668 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_669(self):
        """Test Case 669 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_670(self):
        """Test Case 670 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_671(self):
        """Test Case 671 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_672(self):
        """Test Case 672 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_673(self):
        """Test Case 673 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_674(self):
        """Test Case 674 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_675(self):
        """Test Case 675 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_676(self):
        """Test Case 676 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_677(self):
        """Test Case 677 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_678(self):
        """Test Case 678 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_679(self):
        """Test Case 679 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_680(self):
        """Test Case 680 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_681(self):
        """Test Case 681 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_682(self):
        """Test Case 682 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_683(self):
        """Test Case 683 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_684(self):
        """Test Case 684 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_685(self):
        """Test Case 685 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_686(self):
        """Test Case 686 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_687(self):
        """Test Case 687 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_688(self):
        """Test Case 688 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_689(self):
        """Test Case 689 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_690(self):
        """Test Case 690 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_691(self):
        """Test Case 691 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_692(self):
        """Test Case 692 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_693(self):
        """Test Case 693 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_694(self):
        """Test Case 694 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_695(self):
        """Test Case 695 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_696(self):
        """Test Case 696 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_697(self):
        """Test Case 697 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_698(self):
        """Test Case 698 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_699(self):
        """Test Case 699 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_700(self):
        """Test Case 700 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_701(self):
        """Test Case 701 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_702(self):
        """Test Case 702 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_703(self):
        """Test Case 703 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_704(self):
        """Test Case 704 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_705(self):
        """Test Case 705 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_706(self):
        """Test Case 706 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_707(self):
        """Test Case 707 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_708(self):
        """Test Case 708 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_709(self):
        """Test Case 709 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_710(self):
        """Test Case 710 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_711(self):
        """Test Case 711 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_712(self):
        """Test Case 712 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_713(self):
        """Test Case 713 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_714(self):
        """Test Case 714 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_715(self):
        """Test Case 715 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_716(self):
        """Test Case 716 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_717(self):
        """Test Case 717 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_718(self):
        """Test Case 718 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_719(self):
        """Test Case 719 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_720(self):
        """Test Case 720 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_721(self):
        """Test Case 721 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_722(self):
        """Test Case 722 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_723(self):
        """Test Case 723 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_724(self):
        """Test Case 724 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_725(self):
        """Test Case 725 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_726(self):
        """Test Case 726 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_727(self):
        """Test Case 727 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_728(self):
        """Test Case 728 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_729(self):
        """Test Case 729 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_730(self):
        """Test Case 730 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_731(self):
        """Test Case 731 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_732(self):
        """Test Case 732 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_733(self):
        """Test Case 733 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_734(self):
        """Test Case 734 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_735(self):
        """Test Case 735 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_736(self):
        """Test Case 736 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_737(self):
        """Test Case 737 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_738(self):
        """Test Case 738 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_739(self):
        """Test Case 739 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_740(self):
        """Test Case 740 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_741(self):
        """Test Case 741 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_742(self):
        """Test Case 742 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_743(self):
        """Test Case 743 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_744(self):
        """Test Case 744 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_745(self):
        """Test Case 745 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_746(self):
        """Test Case 746 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.2, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_747(self):
        """Test Case 747 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        dropped = dropout(x, p=0.3, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_748(self):
        """Test Case 748 for custom tensor operations."""
        x = Tensor([0.0, 1.0], requires_grad=True)
        dropped = dropout(x, p=0.4, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_749(self):
        """Test Case 749 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        dropped = dropout(x, p=0.5, training=False)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=1)
        self.assertEqual(len(pooled.data), len(x.data) // 1)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

    def test_tensor_ops_case_750(self):
        """Test Case 750 for custom tensor operations."""
        x = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        dropped = dropout(x, p=0.1, training=True)
        self.assertEqual(len(dropped.data), len(x.data))
        pooled = maxpool2d(x, kernel_size=2)
        self.assertEqual(len(pooled.data), len(x.data) // 2)
        conv = conv2d(x, Tensor([0.5]))
        self.assertEqual(len(conv.data), len(x.data))

if __name__ == '__main__':
    unittest.main()
