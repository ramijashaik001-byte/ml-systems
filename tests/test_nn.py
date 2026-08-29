import unittest
from nexusml.core.tensor import Tensor
from nexusml.core.nn import Linear

class TestNeuralNetwork(unittest.TestCase):

    def test_nn_layer_case_1(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_2(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_3(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_4(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_5(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_6(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_7(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_8(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_9(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_10(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_11(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_12(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_13(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_14(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_15(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_16(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_17(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_18(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_19(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_20(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_21(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_22(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_23(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_24(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_25(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_26(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_27(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_28(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_29(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_30(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_31(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_32(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_33(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_34(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_35(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_36(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_37(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_38(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_39(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_40(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_41(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_42(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_43(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_44(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_45(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_46(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_47(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_48(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_49(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_50(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_51(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_52(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_53(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_54(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_55(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_56(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_57(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_58(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_59(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_60(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_61(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_62(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_63(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_64(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_65(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_66(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_67(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_68(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_69(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_70(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_71(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_72(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_73(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_74(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_75(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_76(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_77(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_78(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_79(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_80(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_81(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_82(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_83(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_84(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_85(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_86(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_87(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_88(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_89(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_90(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_91(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_92(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_93(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_94(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_95(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_96(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_97(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_98(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_99(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_100(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_101(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_102(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_103(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_104(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_105(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_106(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_107(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_108(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_109(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_110(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_111(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_112(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_113(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_114(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_115(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_116(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_117(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_118(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_119(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_120(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_121(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_122(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_123(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_124(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_125(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_126(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_127(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_128(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_129(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_130(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_131(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_132(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_133(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_134(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_135(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_136(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_137(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_138(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_139(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_140(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_141(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_142(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_143(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_144(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_145(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_146(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_147(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_148(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_149(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_150(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_151(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_152(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_153(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_154(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_155(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_156(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_157(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_158(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_159(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_160(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_161(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_162(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_163(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_164(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_165(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_166(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_167(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_168(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_169(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_170(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_171(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_172(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_173(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_174(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_175(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_176(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_177(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_178(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_179(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_180(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_181(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_182(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_183(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_184(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_185(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_186(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_187(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_188(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_189(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_190(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_191(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_192(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_193(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_194(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_195(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_196(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_197(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_198(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_199(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_200(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_201(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_202(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_203(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_204(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_205(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_206(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_207(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_208(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_209(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_210(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_211(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_212(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_213(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_214(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_215(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_216(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_217(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_218(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_219(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_220(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_221(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_222(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_223(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_224(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_225(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_226(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_227(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_228(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_229(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_230(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_231(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_232(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_233(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_234(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_235(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_236(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_237(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_238(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_239(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_240(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_241(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_242(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_243(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_244(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_245(self):
        layer = Linear(4, 2)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_246(self):
        layer = Linear(2, 1)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_247(self):
        layer = Linear(3, 2)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_248(self):
        layer = Linear(4, 1)
        x = Tensor([0.5, 0.5, 0.5, 0.5])
        x.shape = (1, 4)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

    def test_nn_layer_case_249(self):
        layer = Linear(2, 2)
        x = Tensor([0.5, 0.5])
        x.shape = (1, 2)
        out = layer(x)
        self.assertEqual(len(out.data), 2)

    def test_nn_layer_case_250(self):
        layer = Linear(3, 1)
        x = Tensor([0.5, 0.5, 0.5])
        x.shape = (1, 3)
        out = layer(x)
        self.assertEqual(len(out.data), 1)

if __name__ == '__main__':
    unittest.main()
