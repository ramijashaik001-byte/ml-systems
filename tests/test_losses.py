import unittest
from nexusml.core.tensor import Tensor
from nexusml.core.losses import MSELoss, L1Loss

class TestLossFunctions(unittest.TestCase):

    def test_loss_case_1(self):
        pred = Tensor([0.1, 0.2], requires_grad=True)
        target = Tensor([0.1, 0.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_2(self):
        pred = Tensor([0.2, 0.4], requires_grad=True)
        target = Tensor([0.2, 0.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_3(self):
        pred = Tensor([0.30000000000000004, 0.6000000000000001], requires_grad=True)
        target = Tensor([0.30000000000000004, 0.6000000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_4(self):
        pred = Tensor([0.4, 0.8], requires_grad=True)
        target = Tensor([0.4, 0.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_5(self):
        pred = Tensor([0.5, 1.0], requires_grad=True)
        target = Tensor([0.5, 1.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_6(self):
        pred = Tensor([0.6000000000000001, 1.2000000000000002], requires_grad=True)
        target = Tensor([0.6000000000000001, 1.2000000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_7(self):
        pred = Tensor([0.7000000000000001, 1.4000000000000001], requires_grad=True)
        target = Tensor([0.7000000000000001, 1.4000000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_8(self):
        pred = Tensor([0.8, 1.6], requires_grad=True)
        target = Tensor([0.8, 1.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_9(self):
        pred = Tensor([0.9, 1.8], requires_grad=True)
        target = Tensor([0.9, 1.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_10(self):
        pred = Tensor([1.0, 2.0], requires_grad=True)
        target = Tensor([1.0, 2.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_11(self):
        pred = Tensor([1.1, 2.2], requires_grad=True)
        target = Tensor([1.1, 2.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_12(self):
        pred = Tensor([1.2000000000000002, 2.4000000000000004], requires_grad=True)
        target = Tensor([1.2000000000000002, 2.4000000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_13(self):
        pred = Tensor([1.3, 2.6], requires_grad=True)
        target = Tensor([1.3, 2.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_14(self):
        pred = Tensor([1.4000000000000001, 2.8000000000000003], requires_grad=True)
        target = Tensor([1.4000000000000001, 2.8000000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_15(self):
        pred = Tensor([1.5, 3.0], requires_grad=True)
        target = Tensor([1.5, 3.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_16(self):
        pred = Tensor([1.6, 3.2], requires_grad=True)
        target = Tensor([1.6, 3.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_17(self):
        pred = Tensor([1.7000000000000002, 3.4000000000000004], requires_grad=True)
        target = Tensor([1.7000000000000002, 3.4000000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_18(self):
        pred = Tensor([1.8, 3.6], requires_grad=True)
        target = Tensor([1.8, 3.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_19(self):
        pred = Tensor([1.9000000000000001, 3.8000000000000003], requires_grad=True)
        target = Tensor([1.9000000000000001, 3.8000000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_20(self):
        pred = Tensor([2.0, 4.0], requires_grad=True)
        target = Tensor([2.0, 4.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_21(self):
        pred = Tensor([2.1, 4.2], requires_grad=True)
        target = Tensor([2.1, 4.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_22(self):
        pred = Tensor([2.2, 4.4], requires_grad=True)
        target = Tensor([2.2, 4.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_23(self):
        pred = Tensor([2.3000000000000003, 4.6000000000000005], requires_grad=True)
        target = Tensor([2.3000000000000003, 4.6000000000000005])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_24(self):
        pred = Tensor([2.4000000000000004, 4.800000000000001], requires_grad=True)
        target = Tensor([2.4000000000000004, 4.800000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_25(self):
        pred = Tensor([2.5, 5.0], requires_grad=True)
        target = Tensor([2.5, 5.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_26(self):
        pred = Tensor([2.6, 5.2], requires_grad=True)
        target = Tensor([2.6, 5.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_27(self):
        pred = Tensor([2.7, 5.4], requires_grad=True)
        target = Tensor([2.7, 5.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_28(self):
        pred = Tensor([2.8000000000000003, 5.6000000000000005], requires_grad=True)
        target = Tensor([2.8000000000000003, 5.6000000000000005])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_29(self):
        pred = Tensor([2.9000000000000004, 5.800000000000001], requires_grad=True)
        target = Tensor([2.9000000000000004, 5.800000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_30(self):
        pred = Tensor([3.0, 6.0], requires_grad=True)
        target = Tensor([3.0, 6.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_31(self):
        pred = Tensor([3.1, 6.2], requires_grad=True)
        target = Tensor([3.1, 6.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_32(self):
        pred = Tensor([3.2, 6.4], requires_grad=True)
        target = Tensor([3.2, 6.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_33(self):
        pred = Tensor([3.3000000000000003, 6.6000000000000005], requires_grad=True)
        target = Tensor([3.3000000000000003, 6.6000000000000005])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_34(self):
        pred = Tensor([3.4000000000000004, 6.800000000000001], requires_grad=True)
        target = Tensor([3.4000000000000004, 6.800000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_35(self):
        pred = Tensor([3.5, 7.0], requires_grad=True)
        target = Tensor([3.5, 7.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_36(self):
        pred = Tensor([3.6, 7.2], requires_grad=True)
        target = Tensor([3.6, 7.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_37(self):
        pred = Tensor([3.7, 7.4], requires_grad=True)
        target = Tensor([3.7, 7.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_38(self):
        pred = Tensor([3.8000000000000003, 7.6000000000000005], requires_grad=True)
        target = Tensor([3.8000000000000003, 7.6000000000000005])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_39(self):
        pred = Tensor([3.9000000000000004, 7.800000000000001], requires_grad=True)
        target = Tensor([3.9000000000000004, 7.800000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_40(self):
        pred = Tensor([4.0, 8.0], requires_grad=True)
        target = Tensor([4.0, 8.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_41(self):
        pred = Tensor([4.1000000000000005, 8.200000000000001], requires_grad=True)
        target = Tensor([4.1000000000000005, 8.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_42(self):
        pred = Tensor([4.2, 8.4], requires_grad=True)
        target = Tensor([4.2, 8.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_43(self):
        pred = Tensor([4.3, 8.6], requires_grad=True)
        target = Tensor([4.3, 8.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_44(self):
        pred = Tensor([4.4, 8.8], requires_grad=True)
        target = Tensor([4.4, 8.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_45(self):
        pred = Tensor([4.5, 9.0], requires_grad=True)
        target = Tensor([4.5, 9.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_46(self):
        pred = Tensor([4.6000000000000005, 9.200000000000001], requires_grad=True)
        target = Tensor([4.6000000000000005, 9.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_47(self):
        pred = Tensor([4.7, 9.4], requires_grad=True)
        target = Tensor([4.7, 9.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_48(self):
        pred = Tensor([4.800000000000001, 9.600000000000001], requires_grad=True)
        target = Tensor([4.800000000000001, 9.600000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_49(self):
        pred = Tensor([4.9, 9.8], requires_grad=True)
        target = Tensor([4.9, 9.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_50(self):
        pred = Tensor([5.0, 10.0], requires_grad=True)
        target = Tensor([5.0, 10.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_51(self):
        pred = Tensor([5.1000000000000005, 10.200000000000001], requires_grad=True)
        target = Tensor([5.1000000000000005, 10.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_52(self):
        pred = Tensor([5.2, 10.4], requires_grad=True)
        target = Tensor([5.2, 10.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_53(self):
        pred = Tensor([5.300000000000001, 10.600000000000001], requires_grad=True)
        target = Tensor([5.300000000000001, 10.600000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_54(self):
        pred = Tensor([5.4, 10.8], requires_grad=True)
        target = Tensor([5.4, 10.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_55(self):
        pred = Tensor([5.5, 11.0], requires_grad=True)
        target = Tensor([5.5, 11.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_56(self):
        pred = Tensor([5.6000000000000005, 11.200000000000001], requires_grad=True)
        target = Tensor([5.6000000000000005, 11.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_57(self):
        pred = Tensor([5.7, 11.4], requires_grad=True)
        target = Tensor([5.7, 11.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_58(self):
        pred = Tensor([5.800000000000001, 11.600000000000001], requires_grad=True)
        target = Tensor([5.800000000000001, 11.600000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_59(self):
        pred = Tensor([5.9, 11.8], requires_grad=True)
        target = Tensor([5.9, 11.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_60(self):
        pred = Tensor([6.0, 12.0], requires_grad=True)
        target = Tensor([6.0, 12.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_61(self):
        pred = Tensor([6.1000000000000005, 12.200000000000001], requires_grad=True)
        target = Tensor([6.1000000000000005, 12.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_62(self):
        pred = Tensor([6.2, 12.4], requires_grad=True)
        target = Tensor([6.2, 12.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_63(self):
        pred = Tensor([6.300000000000001, 12.600000000000001], requires_grad=True)
        target = Tensor([6.300000000000001, 12.600000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_64(self):
        pred = Tensor([6.4, 12.8], requires_grad=True)
        target = Tensor([6.4, 12.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_65(self):
        pred = Tensor([6.5, 13.0], requires_grad=True)
        target = Tensor([6.5, 13.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_66(self):
        pred = Tensor([6.6000000000000005, 13.200000000000001], requires_grad=True)
        target = Tensor([6.6000000000000005, 13.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_67(self):
        pred = Tensor([6.7, 13.4], requires_grad=True)
        target = Tensor([6.7, 13.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_68(self):
        pred = Tensor([6.800000000000001, 13.600000000000001], requires_grad=True)
        target = Tensor([6.800000000000001, 13.600000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_69(self):
        pred = Tensor([6.9, 13.8], requires_grad=True)
        target = Tensor([6.9, 13.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_70(self):
        pred = Tensor([7.0, 14.0], requires_grad=True)
        target = Tensor([7.0, 14.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_71(self):
        pred = Tensor([7.1000000000000005, 14.200000000000001], requires_grad=True)
        target = Tensor([7.1000000000000005, 14.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_72(self):
        pred = Tensor([7.2, 14.4], requires_grad=True)
        target = Tensor([7.2, 14.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_73(self):
        pred = Tensor([7.300000000000001, 14.600000000000001], requires_grad=True)
        target = Tensor([7.300000000000001, 14.600000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_74(self):
        pred = Tensor([7.4, 14.8], requires_grad=True)
        target = Tensor([7.4, 14.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_75(self):
        pred = Tensor([7.5, 15.0], requires_grad=True)
        target = Tensor([7.5, 15.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_76(self):
        pred = Tensor([7.6000000000000005, 15.200000000000001], requires_grad=True)
        target = Tensor([7.6000000000000005, 15.200000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_77(self):
        pred = Tensor([7.7, 15.4], requires_grad=True)
        target = Tensor([7.7, 15.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_78(self):
        pred = Tensor([7.800000000000001, 15.600000000000001], requires_grad=True)
        target = Tensor([7.800000000000001, 15.600000000000001])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_79(self):
        pred = Tensor([7.9, 15.8], requires_grad=True)
        target = Tensor([7.9, 15.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_80(self):
        pred = Tensor([8.0, 16.0], requires_grad=True)
        target = Tensor([8.0, 16.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_81(self):
        pred = Tensor([8.1, 16.2], requires_grad=True)
        target = Tensor([8.1, 16.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_82(self):
        pred = Tensor([8.200000000000001, 16.400000000000002], requires_grad=True)
        target = Tensor([8.200000000000001, 16.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_83(self):
        pred = Tensor([8.3, 16.6], requires_grad=True)
        target = Tensor([8.3, 16.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_84(self):
        pred = Tensor([8.4, 16.8], requires_grad=True)
        target = Tensor([8.4, 16.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_85(self):
        pred = Tensor([8.5, 17.0], requires_grad=True)
        target = Tensor([8.5, 17.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_86(self):
        pred = Tensor([8.6, 17.2], requires_grad=True)
        target = Tensor([8.6, 17.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_87(self):
        pred = Tensor([8.700000000000001, 17.400000000000002], requires_grad=True)
        target = Tensor([8.700000000000001, 17.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_88(self):
        pred = Tensor([8.8, 17.6], requires_grad=True)
        target = Tensor([8.8, 17.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_89(self):
        pred = Tensor([8.9, 17.8], requires_grad=True)
        target = Tensor([8.9, 17.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_90(self):
        pred = Tensor([9.0, 18.0], requires_grad=True)
        target = Tensor([9.0, 18.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_91(self):
        pred = Tensor([9.1, 18.2], requires_grad=True)
        target = Tensor([9.1, 18.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_92(self):
        pred = Tensor([9.200000000000001, 18.400000000000002], requires_grad=True)
        target = Tensor([9.200000000000001, 18.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_93(self):
        pred = Tensor([9.3, 18.6], requires_grad=True)
        target = Tensor([9.3, 18.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_94(self):
        pred = Tensor([9.4, 18.8], requires_grad=True)
        target = Tensor([9.4, 18.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_95(self):
        pred = Tensor([9.5, 19.0], requires_grad=True)
        target = Tensor([9.5, 19.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_96(self):
        pred = Tensor([9.600000000000001, 19.200000000000003], requires_grad=True)
        target = Tensor([9.600000000000001, 19.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_97(self):
        pred = Tensor([9.700000000000001, 19.400000000000002], requires_grad=True)
        target = Tensor([9.700000000000001, 19.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_98(self):
        pred = Tensor([9.8, 19.6], requires_grad=True)
        target = Tensor([9.8, 19.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_99(self):
        pred = Tensor([9.9, 19.8], requires_grad=True)
        target = Tensor([9.9, 19.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_100(self):
        pred = Tensor([10.0, 20.0], requires_grad=True)
        target = Tensor([10.0, 20.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_101(self):
        pred = Tensor([10.100000000000001, 20.200000000000003], requires_grad=True)
        target = Tensor([10.100000000000001, 20.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_102(self):
        pred = Tensor([10.200000000000001, 20.400000000000002], requires_grad=True)
        target = Tensor([10.200000000000001, 20.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_103(self):
        pred = Tensor([10.3, 20.6], requires_grad=True)
        target = Tensor([10.3, 20.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_104(self):
        pred = Tensor([10.4, 20.8], requires_grad=True)
        target = Tensor([10.4, 20.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_105(self):
        pred = Tensor([10.5, 21.0], requires_grad=True)
        target = Tensor([10.5, 21.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_106(self):
        pred = Tensor([10.600000000000001, 21.200000000000003], requires_grad=True)
        target = Tensor([10.600000000000001, 21.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_107(self):
        pred = Tensor([10.700000000000001, 21.400000000000002], requires_grad=True)
        target = Tensor([10.700000000000001, 21.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_108(self):
        pred = Tensor([10.8, 21.6], requires_grad=True)
        target = Tensor([10.8, 21.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_109(self):
        pred = Tensor([10.9, 21.8], requires_grad=True)
        target = Tensor([10.9, 21.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_110(self):
        pred = Tensor([11.0, 22.0], requires_grad=True)
        target = Tensor([11.0, 22.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_111(self):
        pred = Tensor([11.100000000000001, 22.200000000000003], requires_grad=True)
        target = Tensor([11.100000000000001, 22.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_112(self):
        pred = Tensor([11.200000000000001, 22.400000000000002], requires_grad=True)
        target = Tensor([11.200000000000001, 22.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_113(self):
        pred = Tensor([11.3, 22.6], requires_grad=True)
        target = Tensor([11.3, 22.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_114(self):
        pred = Tensor([11.4, 22.8], requires_grad=True)
        target = Tensor([11.4, 22.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_115(self):
        pred = Tensor([11.5, 23.0], requires_grad=True)
        target = Tensor([11.5, 23.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_116(self):
        pred = Tensor([11.600000000000001, 23.200000000000003], requires_grad=True)
        target = Tensor([11.600000000000001, 23.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_117(self):
        pred = Tensor([11.700000000000001, 23.400000000000002], requires_grad=True)
        target = Tensor([11.700000000000001, 23.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_118(self):
        pred = Tensor([11.8, 23.6], requires_grad=True)
        target = Tensor([11.8, 23.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_119(self):
        pred = Tensor([11.9, 23.8], requires_grad=True)
        target = Tensor([11.9, 23.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_120(self):
        pred = Tensor([12.0, 24.0], requires_grad=True)
        target = Tensor([12.0, 24.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_121(self):
        pred = Tensor([12.100000000000001, 24.200000000000003], requires_grad=True)
        target = Tensor([12.100000000000001, 24.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_122(self):
        pred = Tensor([12.200000000000001, 24.400000000000002], requires_grad=True)
        target = Tensor([12.200000000000001, 24.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_123(self):
        pred = Tensor([12.3, 24.6], requires_grad=True)
        target = Tensor([12.3, 24.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_124(self):
        pred = Tensor([12.4, 24.8], requires_grad=True)
        target = Tensor([12.4, 24.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_125(self):
        pred = Tensor([12.5, 25.0], requires_grad=True)
        target = Tensor([12.5, 25.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_126(self):
        pred = Tensor([12.600000000000001, 25.200000000000003], requires_grad=True)
        target = Tensor([12.600000000000001, 25.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_127(self):
        pred = Tensor([12.700000000000001, 25.400000000000002], requires_grad=True)
        target = Tensor([12.700000000000001, 25.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_128(self):
        pred = Tensor([12.8, 25.6], requires_grad=True)
        target = Tensor([12.8, 25.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_129(self):
        pred = Tensor([12.9, 25.8], requires_grad=True)
        target = Tensor([12.9, 25.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_130(self):
        pred = Tensor([13.0, 26.0], requires_grad=True)
        target = Tensor([13.0, 26.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_131(self):
        pred = Tensor([13.100000000000001, 26.200000000000003], requires_grad=True)
        target = Tensor([13.100000000000001, 26.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_132(self):
        pred = Tensor([13.200000000000001, 26.400000000000002], requires_grad=True)
        target = Tensor([13.200000000000001, 26.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_133(self):
        pred = Tensor([13.3, 26.6], requires_grad=True)
        target = Tensor([13.3, 26.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_134(self):
        pred = Tensor([13.4, 26.8], requires_grad=True)
        target = Tensor([13.4, 26.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_135(self):
        pred = Tensor([13.5, 27.0], requires_grad=True)
        target = Tensor([13.5, 27.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_136(self):
        pred = Tensor([13.600000000000001, 27.200000000000003], requires_grad=True)
        target = Tensor([13.600000000000001, 27.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_137(self):
        pred = Tensor([13.700000000000001, 27.400000000000002], requires_grad=True)
        target = Tensor([13.700000000000001, 27.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_138(self):
        pred = Tensor([13.8, 27.6], requires_grad=True)
        target = Tensor([13.8, 27.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_139(self):
        pred = Tensor([13.9, 27.8], requires_grad=True)
        target = Tensor([13.9, 27.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_140(self):
        pred = Tensor([14.0, 28.0], requires_grad=True)
        target = Tensor([14.0, 28.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_141(self):
        pred = Tensor([14.100000000000001, 28.200000000000003], requires_grad=True)
        target = Tensor([14.100000000000001, 28.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_142(self):
        pred = Tensor([14.200000000000001, 28.400000000000002], requires_grad=True)
        target = Tensor([14.200000000000001, 28.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_143(self):
        pred = Tensor([14.3, 28.6], requires_grad=True)
        target = Tensor([14.3, 28.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_144(self):
        pred = Tensor([14.4, 28.8], requires_grad=True)
        target = Tensor([14.4, 28.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_145(self):
        pred = Tensor([14.5, 29.0], requires_grad=True)
        target = Tensor([14.5, 29.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_146(self):
        pred = Tensor([14.600000000000001, 29.200000000000003], requires_grad=True)
        target = Tensor([14.600000000000001, 29.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_147(self):
        pred = Tensor([14.700000000000001, 29.400000000000002], requires_grad=True)
        target = Tensor([14.700000000000001, 29.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_148(self):
        pred = Tensor([14.8, 29.6], requires_grad=True)
        target = Tensor([14.8, 29.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_149(self):
        pred = Tensor([14.9, 29.8], requires_grad=True)
        target = Tensor([14.9, 29.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_150(self):
        pred = Tensor([15.0, 30.0], requires_grad=True)
        target = Tensor([15.0, 30.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_151(self):
        pred = Tensor([15.100000000000001, 30.200000000000003], requires_grad=True)
        target = Tensor([15.100000000000001, 30.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_152(self):
        pred = Tensor([15.200000000000001, 30.400000000000002], requires_grad=True)
        target = Tensor([15.200000000000001, 30.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_153(self):
        pred = Tensor([15.3, 30.6], requires_grad=True)
        target = Tensor([15.3, 30.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_154(self):
        pred = Tensor([15.4, 30.8], requires_grad=True)
        target = Tensor([15.4, 30.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_155(self):
        pred = Tensor([15.5, 31.0], requires_grad=True)
        target = Tensor([15.5, 31.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_156(self):
        pred = Tensor([15.600000000000001, 31.200000000000003], requires_grad=True)
        target = Tensor([15.600000000000001, 31.200000000000003])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_157(self):
        pred = Tensor([15.700000000000001, 31.400000000000002], requires_grad=True)
        target = Tensor([15.700000000000001, 31.400000000000002])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_158(self):
        pred = Tensor([15.8, 31.6], requires_grad=True)
        target = Tensor([15.8, 31.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_159(self):
        pred = Tensor([15.9, 31.8], requires_grad=True)
        target = Tensor([15.9, 31.8])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_160(self):
        pred = Tensor([16.0, 32.0], requires_grad=True)
        target = Tensor([16.0, 32.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_161(self):
        pred = Tensor([16.1, 32.2], requires_grad=True)
        target = Tensor([16.1, 32.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_162(self):
        pred = Tensor([16.2, 32.4], requires_grad=True)
        target = Tensor([16.2, 32.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_163(self):
        pred = Tensor([16.3, 32.6], requires_grad=True)
        target = Tensor([16.3, 32.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_164(self):
        pred = Tensor([16.400000000000002, 32.800000000000004], requires_grad=True)
        target = Tensor([16.400000000000002, 32.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_165(self):
        pred = Tensor([16.5, 33.0], requires_grad=True)
        target = Tensor([16.5, 33.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_166(self):
        pred = Tensor([16.6, 33.2], requires_grad=True)
        target = Tensor([16.6, 33.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_167(self):
        pred = Tensor([16.7, 33.4], requires_grad=True)
        target = Tensor([16.7, 33.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_168(self):
        pred = Tensor([16.8, 33.6], requires_grad=True)
        target = Tensor([16.8, 33.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_169(self):
        pred = Tensor([16.900000000000002, 33.800000000000004], requires_grad=True)
        target = Tensor([16.900000000000002, 33.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_170(self):
        pred = Tensor([17.0, 34.0], requires_grad=True)
        target = Tensor([17.0, 34.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_171(self):
        pred = Tensor([17.1, 34.2], requires_grad=True)
        target = Tensor([17.1, 34.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_172(self):
        pred = Tensor([17.2, 34.4], requires_grad=True)
        target = Tensor([17.2, 34.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_173(self):
        pred = Tensor([17.3, 34.6], requires_grad=True)
        target = Tensor([17.3, 34.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_174(self):
        pred = Tensor([17.400000000000002, 34.800000000000004], requires_grad=True)
        target = Tensor([17.400000000000002, 34.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_175(self):
        pred = Tensor([17.5, 35.0], requires_grad=True)
        target = Tensor([17.5, 35.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_176(self):
        pred = Tensor([17.6, 35.2], requires_grad=True)
        target = Tensor([17.6, 35.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_177(self):
        pred = Tensor([17.7, 35.4], requires_grad=True)
        target = Tensor([17.7, 35.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_178(self):
        pred = Tensor([17.8, 35.6], requires_grad=True)
        target = Tensor([17.8, 35.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_179(self):
        pred = Tensor([17.900000000000002, 35.800000000000004], requires_grad=True)
        target = Tensor([17.900000000000002, 35.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_180(self):
        pred = Tensor([18.0, 36.0], requires_grad=True)
        target = Tensor([18.0, 36.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_181(self):
        pred = Tensor([18.1, 36.2], requires_grad=True)
        target = Tensor([18.1, 36.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_182(self):
        pred = Tensor([18.2, 36.4], requires_grad=True)
        target = Tensor([18.2, 36.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_183(self):
        pred = Tensor([18.3, 36.6], requires_grad=True)
        target = Tensor([18.3, 36.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_184(self):
        pred = Tensor([18.400000000000002, 36.800000000000004], requires_grad=True)
        target = Tensor([18.400000000000002, 36.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_185(self):
        pred = Tensor([18.5, 37.0], requires_grad=True)
        target = Tensor([18.5, 37.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_186(self):
        pred = Tensor([18.6, 37.2], requires_grad=True)
        target = Tensor([18.6, 37.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_187(self):
        pred = Tensor([18.7, 37.4], requires_grad=True)
        target = Tensor([18.7, 37.4])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_188(self):
        pred = Tensor([18.8, 37.6], requires_grad=True)
        target = Tensor([18.8, 37.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_189(self):
        pred = Tensor([18.900000000000002, 37.800000000000004], requires_grad=True)
        target = Tensor([18.900000000000002, 37.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_190(self):
        pred = Tensor([19.0, 38.0], requires_grad=True)
        target = Tensor([19.0, 38.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_191(self):
        pred = Tensor([19.1, 38.2], requires_grad=True)
        target = Tensor([19.1, 38.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_192(self):
        pred = Tensor([19.200000000000003, 38.400000000000006], requires_grad=True)
        target = Tensor([19.200000000000003, 38.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_193(self):
        pred = Tensor([19.3, 38.6], requires_grad=True)
        target = Tensor([19.3, 38.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_194(self):
        pred = Tensor([19.400000000000002, 38.800000000000004], requires_grad=True)
        target = Tensor([19.400000000000002, 38.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_195(self):
        pred = Tensor([19.5, 39.0], requires_grad=True)
        target = Tensor([19.5, 39.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_196(self):
        pred = Tensor([19.6, 39.2], requires_grad=True)
        target = Tensor([19.6, 39.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_197(self):
        pred = Tensor([19.700000000000003, 39.400000000000006], requires_grad=True)
        target = Tensor([19.700000000000003, 39.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_198(self):
        pred = Tensor([19.8, 39.6], requires_grad=True)
        target = Tensor([19.8, 39.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_199(self):
        pred = Tensor([19.900000000000002, 39.800000000000004], requires_grad=True)
        target = Tensor([19.900000000000002, 39.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_200(self):
        pred = Tensor([20.0, 40.0], requires_grad=True)
        target = Tensor([20.0, 40.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_201(self):
        pred = Tensor([20.1, 40.2], requires_grad=True)
        target = Tensor([20.1, 40.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_202(self):
        pred = Tensor([20.200000000000003, 40.400000000000006], requires_grad=True)
        target = Tensor([20.200000000000003, 40.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_203(self):
        pred = Tensor([20.3, 40.6], requires_grad=True)
        target = Tensor([20.3, 40.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_204(self):
        pred = Tensor([20.400000000000002, 40.800000000000004], requires_grad=True)
        target = Tensor([20.400000000000002, 40.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_205(self):
        pred = Tensor([20.5, 41.0], requires_grad=True)
        target = Tensor([20.5, 41.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_206(self):
        pred = Tensor([20.6, 41.2], requires_grad=True)
        target = Tensor([20.6, 41.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_207(self):
        pred = Tensor([20.700000000000003, 41.400000000000006], requires_grad=True)
        target = Tensor([20.700000000000003, 41.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_208(self):
        pred = Tensor([20.8, 41.6], requires_grad=True)
        target = Tensor([20.8, 41.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_209(self):
        pred = Tensor([20.900000000000002, 41.800000000000004], requires_grad=True)
        target = Tensor([20.900000000000002, 41.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_210(self):
        pred = Tensor([21.0, 42.0], requires_grad=True)
        target = Tensor([21.0, 42.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_211(self):
        pred = Tensor([21.1, 42.2], requires_grad=True)
        target = Tensor([21.1, 42.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_212(self):
        pred = Tensor([21.200000000000003, 42.400000000000006], requires_grad=True)
        target = Tensor([21.200000000000003, 42.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_213(self):
        pred = Tensor([21.3, 42.6], requires_grad=True)
        target = Tensor([21.3, 42.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_214(self):
        pred = Tensor([21.400000000000002, 42.800000000000004], requires_grad=True)
        target = Tensor([21.400000000000002, 42.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_215(self):
        pred = Tensor([21.5, 43.0], requires_grad=True)
        target = Tensor([21.5, 43.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_216(self):
        pred = Tensor([21.6, 43.2], requires_grad=True)
        target = Tensor([21.6, 43.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_217(self):
        pred = Tensor([21.700000000000003, 43.400000000000006], requires_grad=True)
        target = Tensor([21.700000000000003, 43.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_218(self):
        pred = Tensor([21.8, 43.6], requires_grad=True)
        target = Tensor([21.8, 43.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_219(self):
        pred = Tensor([21.900000000000002, 43.800000000000004], requires_grad=True)
        target = Tensor([21.900000000000002, 43.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_220(self):
        pred = Tensor([22.0, 44.0], requires_grad=True)
        target = Tensor([22.0, 44.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_221(self):
        pred = Tensor([22.1, 44.2], requires_grad=True)
        target = Tensor([22.1, 44.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_222(self):
        pred = Tensor([22.200000000000003, 44.400000000000006], requires_grad=True)
        target = Tensor([22.200000000000003, 44.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_223(self):
        pred = Tensor([22.3, 44.6], requires_grad=True)
        target = Tensor([22.3, 44.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_224(self):
        pred = Tensor([22.400000000000002, 44.800000000000004], requires_grad=True)
        target = Tensor([22.400000000000002, 44.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_225(self):
        pred = Tensor([22.5, 45.0], requires_grad=True)
        target = Tensor([22.5, 45.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_226(self):
        pred = Tensor([22.6, 45.2], requires_grad=True)
        target = Tensor([22.6, 45.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_227(self):
        pred = Tensor([22.700000000000003, 45.400000000000006], requires_grad=True)
        target = Tensor([22.700000000000003, 45.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_228(self):
        pred = Tensor([22.8, 45.6], requires_grad=True)
        target = Tensor([22.8, 45.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_229(self):
        pred = Tensor([22.900000000000002, 45.800000000000004], requires_grad=True)
        target = Tensor([22.900000000000002, 45.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_230(self):
        pred = Tensor([23.0, 46.0], requires_grad=True)
        target = Tensor([23.0, 46.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_231(self):
        pred = Tensor([23.1, 46.2], requires_grad=True)
        target = Tensor([23.1, 46.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_232(self):
        pred = Tensor([23.200000000000003, 46.400000000000006], requires_grad=True)
        target = Tensor([23.200000000000003, 46.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_233(self):
        pred = Tensor([23.3, 46.6], requires_grad=True)
        target = Tensor([23.3, 46.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_234(self):
        pred = Tensor([23.400000000000002, 46.800000000000004], requires_grad=True)
        target = Tensor([23.400000000000002, 46.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_235(self):
        pred = Tensor([23.5, 47.0], requires_grad=True)
        target = Tensor([23.5, 47.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_236(self):
        pred = Tensor([23.6, 47.2], requires_grad=True)
        target = Tensor([23.6, 47.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_237(self):
        pred = Tensor([23.700000000000003, 47.400000000000006], requires_grad=True)
        target = Tensor([23.700000000000003, 47.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_238(self):
        pred = Tensor([23.8, 47.6], requires_grad=True)
        target = Tensor([23.8, 47.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_239(self):
        pred = Tensor([23.900000000000002, 47.800000000000004], requires_grad=True)
        target = Tensor([23.900000000000002, 47.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_240(self):
        pred = Tensor([24.0, 48.0], requires_grad=True)
        target = Tensor([24.0, 48.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_241(self):
        pred = Tensor([24.1, 48.2], requires_grad=True)
        target = Tensor([24.1, 48.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_242(self):
        pred = Tensor([24.200000000000003, 48.400000000000006], requires_grad=True)
        target = Tensor([24.200000000000003, 48.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_243(self):
        pred = Tensor([24.3, 48.6], requires_grad=True)
        target = Tensor([24.3, 48.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_244(self):
        pred = Tensor([24.400000000000002, 48.800000000000004], requires_grad=True)
        target = Tensor([24.400000000000002, 48.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_245(self):
        pred = Tensor([24.5, 49.0], requires_grad=True)
        target = Tensor([24.5, 49.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_246(self):
        pred = Tensor([24.6, 49.2], requires_grad=True)
        target = Tensor([24.6, 49.2])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_247(self):
        pred = Tensor([24.700000000000003, 49.400000000000006], requires_grad=True)
        target = Tensor([24.700000000000003, 49.400000000000006])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_248(self):
        pred = Tensor([24.8, 49.6], requires_grad=True)
        target = Tensor([24.8, 49.6])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_249(self):
        pred = Tensor([24.900000000000002, 49.800000000000004], requires_grad=True)
        target = Tensor([24.900000000000002, 49.800000000000004])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_loss_case_250(self):
        pred = Tensor([25.0, 50.0], requires_grad=True)
        target = Tensor([25.0, 50.0])
        mse = MSELoss()
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

if __name__ == '__main__':
    unittest.main()
