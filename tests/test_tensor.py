import unittest
from nexusml.core.tensor import Tensor

class TestTensorAutograd(unittest.TestCase):

    def test_tensor_autograd_case_1(self):
        x = Tensor([0.1, 0.2], requires_grad=True)
        y = Tensor([0.05, 0.15], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 0.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_2(self):
        x = Tensor([0.2, 0.4], requires_grad=True)
        y = Tensor([0.1, 0.3], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 1.05000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_3(self):
        x = Tensor([0.30000000000000004, 0.6000000000000001], requires_grad=True)
        y = Tensor([0.15000000000000002, 0.44999999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 2.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_4(self):
        x = Tensor([0.4, 0.8], requires_grad=True)
        y = Tensor([0.2, 0.6], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 3.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_5(self):
        x = Tensor([0.5, 1.0], requires_grad=True)
        y = Tensor([0.25, 0.75], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 1.12500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_6(self):
        x = Tensor([0.6000000000000001, 1.2000000000000002], requires_grad=True)
        y = Tensor([0.30000000000000004, 0.8999999999999999], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 2.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_7(self):
        x = Tensor([0.7000000000000001, 1.4000000000000001], requires_grad=True)
        y = Tensor([0.35000000000000003, 1.05], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 3.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_8(self):
        x = Tensor([0.8, 1.6], requires_grad=True)
        y = Tensor([0.4, 1.2], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 5.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_9(self):
        x = Tensor([0.9, 1.8], requires_grad=True)
        y = Tensor([0.45, 1.3499999999999999], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 7.65000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_10(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([0.5, 1.5], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 2.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_11(self):
        x = Tensor([1.1, 2.2], requires_grad=True)
        y = Tensor([0.55, 1.65], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 4.12500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_12(self):
        x = Tensor([1.2000000000000002, 2.4000000000000004], requires_grad=True)
        y = Tensor([0.6000000000000001, 1.7999999999999998], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 6.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_13(self):
        x = Tensor([1.3, 2.6], requires_grad=True)
        y = Tensor([0.65, 1.95], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 8.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_14(self):
        x = Tensor([1.4000000000000001, 2.8000000000000003], requires_grad=True)
        y = Tensor([0.7000000000000001, 2.1], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 11.55000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_15(self):
        x = Tensor([1.5, 3.0], requires_grad=True)
        y = Tensor([0.75, 2.25], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 3.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_16(self):
        x = Tensor([1.6, 3.2], requires_grad=True)
        y = Tensor([0.8, 2.4], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 6.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_17(self):
        x = Tensor([1.7000000000000002, 3.4000000000000004], requires_grad=True)
        y = Tensor([0.8500000000000001, 2.55], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 8.92500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_18(self):
        x = Tensor([1.8, 3.6], requires_grad=True)
        y = Tensor([0.9, 2.6999999999999997], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 12.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_19(self):
        x = Tensor([1.9000000000000001, 3.8000000000000003], requires_grad=True)
        y = Tensor([0.9500000000000001, 2.85], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 15.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_20(self):
        x = Tensor([2.0, 4.0], requires_grad=True)
        y = Tensor([1.0, 3.0], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 4.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_21(self):
        x = Tensor([2.1, 4.2], requires_grad=True)
        y = Tensor([1.05, 3.15], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 8.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_22(self):
        x = Tensor([2.2, 4.4], requires_grad=True)
        y = Tensor([1.1, 3.3], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 11.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_23(self):
        x = Tensor([2.3000000000000003, 4.6000000000000005], requires_grad=True)
        y = Tensor([1.1500000000000001, 3.4499999999999997], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 15.52500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_24(self):
        x = Tensor([2.4000000000000004, 4.800000000000001], requires_grad=True)
        y = Tensor([1.2000000000000002, 3.5999999999999996], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 20.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_25(self):
        x = Tensor([2.5, 5.0], requires_grad=True)
        y = Tensor([1.25, 3.75], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 5.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_26(self):
        x = Tensor([2.6, 5.2], requires_grad=True)
        y = Tensor([1.3, 3.9], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 9.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_27(self):
        x = Tensor([2.7, 5.4], requires_grad=True)
        y = Tensor([1.35, 4.05], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 14.85000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_28(self):
        x = Tensor([2.8000000000000003, 5.6000000000000005], requires_grad=True)
        y = Tensor([1.4000000000000001, 4.2], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 19.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_29(self):
        x = Tensor([2.9000000000000004, 5.800000000000001], requires_grad=True)
        y = Tensor([1.4500000000000002, 4.35], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 23.92500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_30(self):
        x = Tensor([3.0, 6.0], requires_grad=True)
        y = Tensor([1.5, 4.5], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 7.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_31(self):
        x = Tensor([3.1, 6.2], requires_grad=True)
        y = Tensor([1.55, 4.6499999999999995], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 11.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_32(self):
        x = Tensor([3.2, 6.4], requires_grad=True)
        y = Tensor([1.6, 4.8], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 16.80000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_33(self):
        x = Tensor([3.3000000000000003, 6.6000000000000005], requires_grad=True)
        y = Tensor([1.6500000000000001, 4.95], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 23.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_34(self):
        x = Tensor([3.4000000000000004, 6.800000000000001], requires_grad=True)
        y = Tensor([1.7000000000000002, 5.1], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 28.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_35(self):
        x = Tensor([3.5, 7.0], requires_grad=True)
        y = Tensor([1.75, 5.25], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 7.87500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_36(self):
        x = Tensor([3.6, 7.2], requires_grad=True)
        y = Tensor([1.8, 5.3999999999999995], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 14.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_37(self):
        x = Tensor([3.7, 7.4], requires_grad=True)
        y = Tensor([1.85, 5.55], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 19.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_38(self):
        x = Tensor([3.8000000000000003, 7.6000000000000005], requires_grad=True)
        y = Tensor([1.9000000000000001, 5.7], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 25.65000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_39(self):
        x = Tensor([3.9000000000000004, 7.800000000000001], requires_grad=True)
        y = Tensor([1.9500000000000002, 5.85], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 33.15000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_40(self):
        x = Tensor([4.0, 8.0], requires_grad=True)
        y = Tensor([2.0, 6.0], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 9.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_41(self):
        x = Tensor([4.1000000000000005, 8.200000000000001], requires_grad=True)
        y = Tensor([2.0500000000000003, 6.1499999999999995], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 15.37500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_42(self):
        x = Tensor([4.2, 8.4], requires_grad=True)
        y = Tensor([2.1, 6.3], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 23.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_43(self):
        x = Tensor([4.3, 8.6], requires_grad=True)
        y = Tensor([2.15, 6.45], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 29.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_44(self):
        x = Tensor([4.4, 8.8], requires_grad=True)
        y = Tensor([2.2, 6.6], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 36.30000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_45(self):
        x = Tensor([4.5, 9.0], requires_grad=True)
        y = Tensor([2.25, 6.75], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 11.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_46(self):
        x = Tensor([4.6000000000000005, 9.200000000000001], requires_grad=True)
        y = Tensor([2.3000000000000003, 6.8999999999999995], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 17.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_47(self):
        x = Tensor([4.7, 9.4], requires_grad=True)
        y = Tensor([2.35, 7.05], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 24.67500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_48(self):
        x = Tensor([4.800000000000001, 9.600000000000001], requires_grad=True)
        y = Tensor([2.4000000000000004, 7.199999999999999], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 33.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_49(self):
        x = Tensor([4.9, 9.8], requires_grad=True)
        y = Tensor([2.45, 7.35], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 40.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_50(self):
        x = Tensor([5.0, 10.0], requires_grad=True)
        y = Tensor([2.5, 7.5], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 11.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_51(self):
        x = Tensor([5.1000000000000005, 10.200000000000001], requires_grad=True)
        y = Tensor([2.5500000000000003, 7.6499999999999995], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 20.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_52(self):
        x = Tensor([5.2, 10.4], requires_grad=True)
        y = Tensor([2.6, 7.8], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 27.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_53(self):
        x = Tensor([5.300000000000001, 10.600000000000001], requires_grad=True)
        y = Tensor([2.6500000000000004, 7.949999999999999], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 35.77500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_54(self):
        x = Tensor([5.4, 10.8], requires_grad=True)
        y = Tensor([2.7, 8.1], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 45.90000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_55(self):
        x = Tensor([5.5, 11.0], requires_grad=True)
        y = Tensor([2.75, 8.25], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 12.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_56(self):
        x = Tensor([5.6000000000000005, 11.200000000000001], requires_grad=True)
        y = Tensor([2.8000000000000003, 8.4], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 21.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_57(self):
        x = Tensor([5.7, 11.4], requires_grad=True)
        y = Tensor([2.85, 8.549999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 31.35000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_58(self):
        x = Tensor([5.800000000000001, 11.600000000000001], requires_grad=True)
        y = Tensor([2.9000000000000004, 8.7], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 39.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_59(self):
        x = Tensor([5.9, 11.8], requires_grad=True)
        y = Tensor([2.95, 8.85], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 48.67500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_60(self):
        x = Tensor([6.0, 12.0], requires_grad=True)
        y = Tensor([3.0, 9.0], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 15.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_61(self):
        x = Tensor([6.1000000000000005, 12.200000000000001], requires_grad=True)
        y = Tensor([3.0500000000000003, 9.15], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 23.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_62(self):
        x = Tensor([6.2, 12.4], requires_grad=True)
        y = Tensor([3.1, 9.299999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 32.55000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_63(self):
        x = Tensor([6.300000000000001, 12.600000000000001], requires_grad=True)
        y = Tensor([3.1500000000000004, 9.45], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 44.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_64(self):
        x = Tensor([6.4, 12.8], requires_grad=True)
        y = Tensor([3.2, 9.6], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 53.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_65(self):
        x = Tensor([6.5, 13.0], requires_grad=True)
        y = Tensor([3.25, 9.75], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 14.62500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_66(self):
        x = Tensor([6.6000000000000005, 13.200000000000001], requires_grad=True)
        y = Tensor([3.3000000000000003, 9.9], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 26.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_67(self):
        x = Tensor([6.7, 13.4], requires_grad=True)
        y = Tensor([3.35, 10.049999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 35.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_68(self):
        x = Tensor([6.800000000000001, 13.600000000000001], requires_grad=True)
        y = Tensor([3.4000000000000004, 10.2], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 45.90000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_69(self):
        x = Tensor([6.9, 13.8], requires_grad=True)
        y = Tensor([3.45, 10.35], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 58.65000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_70(self):
        x = Tensor([7.0, 14.0], requires_grad=True)
        y = Tensor([3.5, 10.5], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 16.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_71(self):
        x = Tensor([7.1000000000000005, 14.200000000000001], requires_grad=True)
        y = Tensor([3.5500000000000003, 10.65], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 26.62500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_72(self):
        x = Tensor([7.2, 14.4], requires_grad=True)
        y = Tensor([3.6, 10.799999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 39.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_73(self):
        x = Tensor([7.300000000000001, 14.600000000000001], requires_grad=True)
        y = Tensor([3.6500000000000004, 10.95], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 49.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_74(self):
        x = Tensor([7.4, 14.8], requires_grad=True)
        y = Tensor([3.7, 11.1], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 61.05000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_75(self):
        x = Tensor([7.5, 15.0], requires_grad=True)
        y = Tensor([3.75, 11.25], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 18.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_76(self):
        x = Tensor([7.6000000000000005, 15.200000000000001], requires_grad=True)
        y = Tensor([3.8000000000000003, 11.4], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 29.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_77(self):
        x = Tensor([7.7, 15.4], requires_grad=True)
        y = Tensor([3.85, 11.549999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 40.42500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_78(self):
        x = Tensor([7.800000000000001, 15.600000000000001], requires_grad=True)
        y = Tensor([3.9000000000000004, 11.7], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 54.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_79(self):
        x = Tensor([7.9, 15.8], requires_grad=True)
        y = Tensor([3.95, 11.85], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 65.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_80(self):
        x = Tensor([8.0, 16.0], requires_grad=True)
        y = Tensor([4.0, 12.0], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 18.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_81(self):
        x = Tensor([8.1, 16.2], requires_grad=True)
        y = Tensor([4.05, 12.15], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 32.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_82(self):
        x = Tensor([8.200000000000001, 16.400000000000002], requires_grad=True)
        y = Tensor([4.1000000000000005, 12.299999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 43.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_83(self):
        x = Tensor([8.3, 16.6], requires_grad=True)
        y = Tensor([4.15, 12.45], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 56.02500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_84(self):
        x = Tensor([8.4, 16.8], requires_grad=True)
        y = Tensor([4.2, 12.6], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 71.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_85(self):
        x = Tensor([8.5, 17.0], requires_grad=True)
        y = Tensor([4.25, 12.75], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 19.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_86(self):
        x = Tensor([8.6, 17.2], requires_grad=True)
        y = Tensor([4.3, 12.9], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 32.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_87(self):
        x = Tensor([8.700000000000001, 17.400000000000002], requires_grad=True)
        y = Tensor([4.3500000000000005, 13.049999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 47.85000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_88(self):
        x = Tensor([8.8, 17.6], requires_grad=True)
        y = Tensor([4.4, 13.2], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 60.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_89(self):
        x = Tensor([8.9, 17.8], requires_grad=True)
        y = Tensor([4.45, 13.35], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 73.42500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_90(self):
        x = Tensor([9.0, 18.0], requires_grad=True)
        y = Tensor([4.5, 13.5], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 22.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_91(self):
        x = Tensor([9.1, 18.2], requires_grad=True)
        y = Tensor([4.55, 13.65], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 34.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_92(self):
        x = Tensor([9.200000000000001, 18.400000000000002], requires_grad=True)
        y = Tensor([4.6000000000000005, 13.799999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 48.30000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_93(self):
        x = Tensor([9.3, 18.6], requires_grad=True)
        y = Tensor([4.65, 13.95], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 65.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_94(self):
        x = Tensor([9.4, 18.8], requires_grad=True)
        y = Tensor([4.7, 14.1], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 78.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_95(self):
        x = Tensor([9.5, 19.0], requires_grad=True)
        y = Tensor([4.75, 14.25], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 21.37500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_96(self):
        x = Tensor([9.600000000000001, 19.200000000000003], requires_grad=True)
        y = Tensor([4.800000000000001, 14.399999999999999], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 38.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_97(self):
        x = Tensor([9.700000000000001, 19.400000000000002], requires_grad=True)
        y = Tensor([4.8500000000000005, 14.549999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 51.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_98(self):
        x = Tensor([9.8, 19.6], requires_grad=True)
        y = Tensor([4.9, 14.7], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 66.15000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_99(self):
        x = Tensor([9.9, 19.8], requires_grad=True)
        y = Tensor([4.95, 14.85], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 84.15000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_100(self):
        x = Tensor([10.0, 20.0], requires_grad=True)
        y = Tensor([5.0, 15.0], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 23.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_101(self):
        x = Tensor([10.100000000000001, 20.200000000000003], requires_grad=True)
        y = Tensor([5.050000000000001, 15.149999999999999], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 37.87500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_102(self):
        x = Tensor([10.200000000000001, 20.400000000000002], requires_grad=True)
        y = Tensor([5.1000000000000005, 15.299999999999999], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 56.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_103(self):
        x = Tensor([10.3, 20.6], requires_grad=True)
        y = Tensor([5.15, 15.45], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 70.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_104(self):
        x = Tensor([10.4, 20.8], requires_grad=True)
        y = Tensor([5.2, 15.6], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 85.80000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_105(self):
        x = Tensor([10.5, 21.0], requires_grad=True)
        y = Tensor([5.25, 15.75], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 26.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_106(self):
        x = Tensor([10.600000000000001, 21.200000000000003], requires_grad=True)
        y = Tensor([5.300000000000001, 15.899999999999999], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 40.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_107(self):
        x = Tensor([10.700000000000001, 21.400000000000002], requires_grad=True)
        y = Tensor([5.3500000000000005, 16.05], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 56.17500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_108(self):
        x = Tensor([10.8, 21.6], requires_grad=True)
        y = Tensor([5.4, 16.2], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 75.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_109(self):
        x = Tensor([10.9, 21.8], requires_grad=True)
        y = Tensor([5.45, 16.349999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 90.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_110(self):
        x = Tensor([11.0, 22.0], requires_grad=True)
        y = Tensor([5.5, 16.5], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 24.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_111(self):
        x = Tensor([11.100000000000001, 22.200000000000003], requires_grad=True)
        y = Tensor([5.550000000000001, 16.65], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 44.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_112(self):
        x = Tensor([11.200000000000001, 22.400000000000002], requires_grad=True)
        y = Tensor([5.6000000000000005, 16.8], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 59.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_113(self):
        x = Tensor([11.3, 22.6], requires_grad=True)
        y = Tensor([5.65, 16.95], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 76.27500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_114(self):
        x = Tensor([11.4, 22.8], requires_grad=True)
        y = Tensor([5.7, 17.099999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 96.90000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_115(self):
        x = Tensor([11.5, 23.0], requires_grad=True)
        y = Tensor([5.75, 17.25], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 26.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_116(self):
        x = Tensor([11.600000000000001, 23.200000000000003], requires_grad=True)
        y = Tensor([5.800000000000001, 17.4], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 43.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_117(self):
        x = Tensor([11.700000000000001, 23.400000000000002], requires_grad=True)
        y = Tensor([5.8500000000000005, 17.55], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 64.35000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_118(self):
        x = Tensor([11.8, 23.6], requires_grad=True)
        y = Tensor([5.9, 17.7], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 80.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_119(self):
        x = Tensor([11.9, 23.8], requires_grad=True)
        y = Tensor([5.95, 17.849999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 98.17500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_120(self):
        x = Tensor([12.0, 24.0], requires_grad=True)
        y = Tensor([6.0, 18.0], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 30.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_121(self):
        x = Tensor([12.100000000000001, 24.200000000000003], requires_grad=True)
        y = Tensor([6.050000000000001, 18.15], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 46.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_122(self):
        x = Tensor([12.200000000000001, 24.400000000000002], requires_grad=True)
        y = Tensor([6.1000000000000005, 18.3], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 64.05000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_123(self):
        x = Tensor([12.3, 24.6], requires_grad=True)
        y = Tensor([6.15, 18.45], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 86.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_124(self):
        x = Tensor([12.4, 24.8], requires_grad=True)
        y = Tensor([6.2, 18.599999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 103.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_125(self):
        x = Tensor([12.5, 25.0], requires_grad=True)
        y = Tensor([6.25, 18.75], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 28.12500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_126(self):
        x = Tensor([12.600000000000001, 25.200000000000003], requires_grad=True)
        y = Tensor([6.300000000000001, 18.9], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 50.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_127(self):
        x = Tensor([12.700000000000001, 25.400000000000002], requires_grad=True)
        y = Tensor([6.3500000000000005, 19.05], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 67.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_128(self):
        x = Tensor([12.8, 25.6], requires_grad=True)
        y = Tensor([6.4, 19.2], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 86.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_129(self):
        x = Tensor([12.9, 25.8], requires_grad=True)
        y = Tensor([6.45, 19.349999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 109.65000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_130(self):
        x = Tensor([13.0, 26.0], requires_grad=True)
        y = Tensor([6.5, 19.5], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 30.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_131(self):
        x = Tensor([13.100000000000001, 26.200000000000003], requires_grad=True)
        y = Tensor([6.550000000000001, 19.65], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 49.12500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_132(self):
        x = Tensor([13.200000000000001, 26.400000000000002], requires_grad=True)
        y = Tensor([6.6000000000000005, 19.8], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 72.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_133(self):
        x = Tensor([13.3, 26.6], requires_grad=True)
        y = Tensor([6.65, 19.95], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 90.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_134(self):
        x = Tensor([13.4, 26.8], requires_grad=True)
        y = Tensor([6.7, 20.099999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 110.55000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_135(self):
        x = Tensor([13.5, 27.0], requires_grad=True)
        y = Tensor([6.75, 20.25], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 33.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_136(self):
        x = Tensor([13.600000000000001, 27.200000000000003], requires_grad=True)
        y = Tensor([6.800000000000001, 20.4], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 52.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_137(self):
        x = Tensor([13.700000000000001, 27.400000000000002], requires_grad=True)
        y = Tensor([6.8500000000000005, 20.55], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 71.92500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_138(self):
        x = Tensor([13.8, 27.6], requires_grad=True)
        y = Tensor([6.9, 20.7], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 96.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_139(self):
        x = Tensor([13.9, 27.8], requires_grad=True)
        y = Tensor([6.95, 20.849999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 115.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_140(self):
        x = Tensor([14.0, 28.0], requires_grad=True)
        y = Tensor([7.0, 21.0], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 31.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_141(self):
        x = Tensor([14.100000000000001, 28.200000000000003], requires_grad=True)
        y = Tensor([7.050000000000001, 21.15], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 56.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_142(self):
        x = Tensor([14.200000000000001, 28.400000000000002], requires_grad=True)
        y = Tensor([7.1000000000000005, 21.3], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 75.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_143(self):
        x = Tensor([14.3, 28.6], requires_grad=True)
        y = Tensor([7.15, 21.45], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 96.52500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_144(self):
        x = Tensor([14.4, 28.8], requires_grad=True)
        y = Tensor([7.2, 21.599999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 122.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_145(self):
        x = Tensor([14.5, 29.0], requires_grad=True)
        y = Tensor([7.25, 21.75], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 33.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_146(self):
        x = Tensor([14.600000000000001, 29.200000000000003], requires_grad=True)
        y = Tensor([7.300000000000001, 21.9], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 54.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_147(self):
        x = Tensor([14.700000000000001, 29.400000000000002], requires_grad=True)
        y = Tensor([7.3500000000000005, 22.05], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 80.85000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_148(self):
        x = Tensor([14.8, 29.6], requires_grad=True)
        y = Tensor([7.4, 22.2], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 101.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_149(self):
        x = Tensor([14.9, 29.8], requires_grad=True)
        y = Tensor([7.45, 22.349999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 122.92500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_150(self):
        x = Tensor([15.0, 30.0], requires_grad=True)
        y = Tensor([7.5, 22.5], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 37.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_151(self):
        x = Tensor([15.100000000000001, 30.200000000000003], requires_grad=True)
        y = Tensor([7.550000000000001, 22.65], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 57.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_152(self):
        x = Tensor([15.200000000000001, 30.400000000000002], requires_grad=True)
        y = Tensor([7.6000000000000005, 22.8], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 79.80000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_153(self):
        x = Tensor([15.3, 30.6], requires_grad=True)
        y = Tensor([7.65, 22.95], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 107.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_154(self):
        x = Tensor([15.4, 30.8], requires_grad=True)
        y = Tensor([7.7, 23.099999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 128.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_155(self):
        x = Tensor([15.5, 31.0], requires_grad=True)
        y = Tensor([7.75, 23.25], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 34.87500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_156(self):
        x = Tensor([15.600000000000001, 31.200000000000003], requires_grad=True)
        y = Tensor([7.800000000000001, 23.4], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 62.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_157(self):
        x = Tensor([15.700000000000001, 31.400000000000002], requires_grad=True)
        y = Tensor([7.8500000000000005, 23.55], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 83.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_158(self):
        x = Tensor([15.8, 31.6], requires_grad=True)
        y = Tensor([7.9, 23.7], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 106.65000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_159(self):
        x = Tensor([15.9, 31.8], requires_grad=True)
        y = Tensor([7.95, 23.849999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 135.15000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_160(self):
        x = Tensor([16.0, 32.0], requires_grad=True)
        y = Tensor([8.0, 24.0], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 37.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_161(self):
        x = Tensor([16.1, 32.2], requires_grad=True)
        y = Tensor([8.05, 24.15], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 60.37500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_162(self):
        x = Tensor([16.2, 32.4], requires_grad=True)
        y = Tensor([8.1, 24.3], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 89.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_163(self):
        x = Tensor([16.3, 32.6], requires_grad=True)
        y = Tensor([8.15, 24.45], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 111.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_164(self):
        x = Tensor([16.400000000000002, 32.800000000000004], requires_grad=True)
        y = Tensor([8.200000000000001, 24.599999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 135.30000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_165(self):
        x = Tensor([16.5, 33.0], requires_grad=True)
        y = Tensor([8.25, 24.75], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 41.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_166(self):
        x = Tensor([16.6, 33.2], requires_grad=True)
        y = Tensor([8.3, 24.9], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 63.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_167(self):
        x = Tensor([16.7, 33.4], requires_grad=True)
        y = Tensor([8.35, 25.05], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 87.67500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_168(self):
        x = Tensor([16.8, 33.6], requires_grad=True)
        y = Tensor([8.4, 25.2], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 117.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_169(self):
        x = Tensor([16.900000000000002, 33.800000000000004], requires_grad=True)
        y = Tensor([8.450000000000001, 25.349999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 140.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_170(self):
        x = Tensor([17.0, 34.0], requires_grad=True)
        y = Tensor([8.5, 25.5], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 38.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_171(self):
        x = Tensor([17.1, 34.2], requires_grad=True)
        y = Tensor([8.55, 25.65], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 68.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_172(self):
        x = Tensor([17.2, 34.4], requires_grad=True)
        y = Tensor([8.6, 25.8], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 91.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_173(self):
        x = Tensor([17.3, 34.6], requires_grad=True)
        y = Tensor([8.65, 25.95], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 116.77500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_174(self):
        x = Tensor([17.400000000000002, 34.800000000000004], requires_grad=True)
        y = Tensor([8.700000000000001, 26.099999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 147.90000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_175(self):
        x = Tensor([17.5, 35.0], requires_grad=True)
        y = Tensor([8.75, 26.25], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 40.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_176(self):
        x = Tensor([17.6, 35.2], requires_grad=True)
        y = Tensor([8.8, 26.4], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 66.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_177(self):
        x = Tensor([17.7, 35.4], requires_grad=True)
        y = Tensor([8.85, 26.55], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 97.35000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_178(self):
        x = Tensor([17.8, 35.6], requires_grad=True)
        y = Tensor([8.9, 26.7], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 121.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_179(self):
        x = Tensor([17.900000000000002, 35.800000000000004], requires_grad=True)
        y = Tensor([8.950000000000001, 26.849999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 147.67500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_180(self):
        x = Tensor([18.0, 36.0], requires_grad=True)
        y = Tensor([9.0, 27.0], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 45.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_181(self):
        x = Tensor([18.1, 36.2], requires_grad=True)
        y = Tensor([9.05, 27.15], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 69.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_182(self):
        x = Tensor([18.2, 36.4], requires_grad=True)
        y = Tensor([9.1, 27.3], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 95.55000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_183(self):
        x = Tensor([18.3, 36.6], requires_grad=True)
        y = Tensor([9.15, 27.45], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 128.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_184(self):
        x = Tensor([18.400000000000002, 36.800000000000004], requires_grad=True)
        y = Tensor([9.200000000000001, 27.599999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 153.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_185(self):
        x = Tensor([18.5, 37.0], requires_grad=True)
        y = Tensor([9.25, 27.75], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 41.62500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_186(self):
        x = Tensor([18.6, 37.2], requires_grad=True)
        y = Tensor([9.3, 27.9], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 74.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_187(self):
        x = Tensor([18.7, 37.4], requires_grad=True)
        y = Tensor([9.35, 28.05], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 99.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_188(self):
        x = Tensor([18.8, 37.6], requires_grad=True)
        y = Tensor([9.4, 28.2], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 126.90000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_189(self):
        x = Tensor([18.900000000000002, 37.800000000000004], requires_grad=True)
        y = Tensor([9.450000000000001, 28.349999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 160.65000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_190(self):
        x = Tensor([19.0, 38.0], requires_grad=True)
        y = Tensor([9.5, 28.5], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 44.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_191(self):
        x = Tensor([19.1, 38.2], requires_grad=True)
        y = Tensor([9.55, 28.65], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 71.62500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_192(self):
        x = Tensor([19.200000000000003, 38.400000000000006], requires_grad=True)
        y = Tensor([9.600000000000001, 28.799999999999997], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 105.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_193(self):
        x = Tensor([19.3, 38.6], requires_grad=True)
        y = Tensor([9.65, 28.95], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 131.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_194(self):
        x = Tensor([19.400000000000002, 38.800000000000004], requires_grad=True)
        y = Tensor([9.700000000000001, 29.099999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 160.05000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_195(self):
        x = Tensor([19.5, 39.0], requires_grad=True)
        y = Tensor([9.75, 29.25], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 48.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_196(self):
        x = Tensor([19.6, 39.2], requires_grad=True)
        y = Tensor([9.8, 29.4], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 75.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_197(self):
        x = Tensor([19.700000000000003, 39.400000000000006], requires_grad=True)
        y = Tensor([9.850000000000001, 29.549999999999997], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 103.42500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_198(self):
        x = Tensor([19.8, 39.6], requires_grad=True)
        y = Tensor([9.9, 29.7], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 138.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_199(self):
        x = Tensor([19.900000000000002, 39.800000000000004], requires_grad=True)
        y = Tensor([9.950000000000001, 29.849999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 165.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_200(self):
        x = Tensor([20.0, 40.0], requires_grad=True)
        y = Tensor([10.0, 30.0], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 45.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_201(self):
        x = Tensor([20.1, 40.2], requires_grad=True)
        y = Tensor([10.05, 30.15], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 80.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_202(self):
        x = Tensor([20.200000000000003, 40.400000000000006], requires_grad=True)
        y = Tensor([10.100000000000001, 30.299999999999997], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 107.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_203(self):
        x = Tensor([20.3, 40.6], requires_grad=True)
        y = Tensor([10.15, 30.45], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 137.02500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_204(self):
        x = Tensor([20.400000000000002, 40.800000000000004], requires_grad=True)
        y = Tensor([10.200000000000001, 30.599999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 173.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_205(self):
        x = Tensor([20.5, 41.0], requires_grad=True)
        y = Tensor([10.25, 30.75], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 47.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_206(self):
        x = Tensor([20.6, 41.2], requires_grad=True)
        y = Tensor([10.3, 30.9], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 77.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_207(self):
        x = Tensor([20.700000000000003, 41.400000000000006], requires_grad=True)
        y = Tensor([10.350000000000001, 31.049999999999997], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 113.85000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_208(self):
        x = Tensor([20.8, 41.6], requires_grad=True)
        y = Tensor([10.4, 31.2], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 142.13333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_209(self):
        x = Tensor([20.900000000000002, 41.800000000000004], requires_grad=True)
        y = Tensor([10.450000000000001, 31.349999999999998], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 172.42500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_210(self):
        x = Tensor([21.0, 42.0], requires_grad=True)
        y = Tensor([10.5, 31.5], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 52.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_211(self):
        x = Tensor([21.1, 42.2], requires_grad=True)
        y = Tensor([10.55, 31.65], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 80.88333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_212(self):
        x = Tensor([21.200000000000003, 42.400000000000006], requires_grad=True)
        y = Tensor([10.600000000000001, 31.799999999999997], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 111.30000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_213(self):
        x = Tensor([21.3, 42.6], requires_grad=True)
        y = Tensor([10.65, 31.95], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 149.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_214(self):
        x = Tensor([21.400000000000002, 42.800000000000004], requires_grad=True)
        y = Tensor([10.700000000000001, 32.1], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 178.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_215(self):
        x = Tensor([21.5, 43.0], requires_grad=True)
        y = Tensor([10.75, 32.25], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 48.37500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_216(self):
        x = Tensor([21.6, 43.2], requires_grad=True)
        y = Tensor([10.8, 32.4], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 86.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_217(self):
        x = Tensor([21.700000000000003, 43.400000000000006], requires_grad=True)
        y = Tensor([10.850000000000001, 32.55], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 115.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_218(self):
        x = Tensor([21.8, 43.6], requires_grad=True)
        y = Tensor([10.9, 32.699999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 147.15000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_219(self):
        x = Tensor([21.900000000000002, 43.800000000000004], requires_grad=True)
        y = Tensor([10.950000000000001, 32.85], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 186.15000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_220(self):
        x = Tensor([22.0, 44.0], requires_grad=True)
        y = Tensor([11.0, 33.0], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 51.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_221(self):
        x = Tensor([22.1, 44.2], requires_grad=True)
        y = Tensor([11.05, 33.15], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 82.87500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_222(self):
        x = Tensor([22.200000000000003, 44.400000000000006], requires_grad=True)
        y = Tensor([11.100000000000001, 33.3], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 122.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_223(self):
        x = Tensor([22.3, 44.6], requires_grad=True)
        y = Tensor([11.15, 33.449999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 152.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_224(self):
        x = Tensor([22.400000000000002, 44.800000000000004], requires_grad=True)
        y = Tensor([11.200000000000001, 33.6], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 184.80000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_225(self):
        x = Tensor([22.5, 45.0], requires_grad=True)
        y = Tensor([11.25, 33.75], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 56.25000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_226(self):
        x = Tensor([22.6, 45.2], requires_grad=True)
        y = Tensor([11.3, 33.9], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 86.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_227(self):
        x = Tensor([22.700000000000003, 45.400000000000006], requires_grad=True)
        y = Tensor([11.350000000000001, 34.05], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 119.17500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_228(self):
        x = Tensor([22.8, 45.6], requires_grad=True)
        y = Tensor([11.4, 34.199999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 159.60000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_229(self):
        x = Tensor([22.900000000000002, 45.800000000000004], requires_grad=True)
        y = Tensor([11.450000000000001, 34.35], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 190.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_230(self):
        x = Tensor([23.0, 46.0], requires_grad=True)
        y = Tensor([11.5, 34.5], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 51.75000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_231(self):
        x = Tensor([23.1, 46.2], requires_grad=True)
        y = Tensor([11.55, 34.65], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 92.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_232(self):
        x = Tensor([23.200000000000003, 46.400000000000006], requires_grad=True)
        y = Tensor([11.600000000000001, 34.8], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 123.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_233(self):
        x = Tensor([23.3, 46.6], requires_grad=True)
        y = Tensor([11.65, 34.949999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 157.27500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_234(self):
        x = Tensor([23.400000000000002, 46.800000000000004], requires_grad=True)
        y = Tensor([11.700000000000001, 35.1], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 198.90000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_235(self):
        x = Tensor([23.5, 47.0], requires_grad=True)
        y = Tensor([11.75, 35.25], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 54.83333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_236(self):
        x = Tensor([23.6, 47.2], requires_grad=True)
        y = Tensor([11.8, 35.4], requires_grad=True)
        z = (x * 2.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 88.50000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_237(self):
        x = Tensor([23.700000000000003, 47.400000000000006], requires_grad=True)
        y = Tensor([11.850000000000001, 35.55], requires_grad=True)
        z = (x * 3.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 130.35000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_238(self):
        x = Tensor([23.8, 47.6], requires_grad=True)
        y = Tensor([11.9, 35.699999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 162.63333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_239(self):
        x = Tensor([23.900000000000002, 47.800000000000004], requires_grad=True)
        y = Tensor([11.950000000000001, 35.85], requires_grad=True)
        z = (x * 5.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 197.17500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_240(self):
        x = Tensor([24.0, 48.0], requires_grad=True)
        y = Tensor([12.0, 36.0], requires_grad=True)
        z = (x * 1.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 60.00000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_241(self):
        x = Tensor([24.1, 48.2], requires_grad=True)
        y = Tensor([12.05, 36.15], requires_grad=True)
        z = (x * 2.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 92.38333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_242(self):
        x = Tensor([24.200000000000003, 48.400000000000006], requires_grad=True)
        y = Tensor([12.100000000000001, 36.3], requires_grad=True)
        z = (x * 3.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 127.05000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_243(self):
        x = Tensor([24.3, 48.6], requires_grad=True)
        y = Tensor([12.15, 36.449999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 170.10000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_244(self):
        x = Tensor([24.400000000000002, 48.800000000000004], requires_grad=True)
        y = Tensor([12.200000000000001, 36.6], requires_grad=True)
        z = (x * 5.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 203.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_245(self):
        x = Tensor([24.5, 49.0], requires_grad=True)
        y = Tensor([12.25, 36.75], requires_grad=True)
        z = (x * 1.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 55.12500, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_246(self):
        x = Tensor([24.6, 49.2], requires_grad=True)
        y = Tensor([12.3, 36.9], requires_grad=True)
        z = (x * 2.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 98.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_247(self):
        x = Tensor([24.700000000000003, 49.400000000000006], requires_grad=True)
        y = Tensor([12.350000000000001, 37.05], requires_grad=True)
        z = (x * 3.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 131.73333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_248(self):
        x = Tensor([24.8, 49.6], requires_grad=True)
        y = Tensor([12.4, 37.199999999999996], requires_grad=True)
        z = (x * 4.0) + (y / 4.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 167.40000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_249(self):
        x = Tensor([24.900000000000002, 49.800000000000004], requires_grad=True)
        y = Tensor([12.450000000000001, 37.35], requires_grad=True)
        z = (x * 5.0) + (y / 2.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 211.65000, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_tensor_autograd_case_250(self):
        x = Tensor([25.0, 50.0], requires_grad=True)
        y = Tensor([12.5, 37.5], requires_grad=True)
        z = (x * 1.0) + (y / 3.0) + (x - y)
        result = z.mean()
        result.backward()
        self.assertAlmostEqual(result.data[0], 58.33333, places=4)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

if __name__ == '__main__':
    unittest.main()
