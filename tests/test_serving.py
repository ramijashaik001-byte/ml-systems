import unittest
from nexusml.serving.engine import InferenceCache, ABRouter

class TestModelServing(unittest.TestCase):

    def test_serving_case_1(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_2(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_3(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_4(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_5(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_6(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_7(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_8(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_9(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_10(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_11(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_12(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_13(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_14(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_15(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_16(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_17(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_18(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_19(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_20(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_21(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_22(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_23(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_24(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_25(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_26(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_27(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_28(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_29(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_30(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_31(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_32(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_33(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_34(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_35(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_36(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_37(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_38(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_39(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_40(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_41(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_42(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_43(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_44(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_45(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_46(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_47(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_48(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_49(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_50(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_51(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_52(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_53(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_54(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_55(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_56(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_57(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_58(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_59(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_60(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_61(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_62(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_63(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_64(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_65(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_66(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_67(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_68(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_69(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_70(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_71(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_72(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_73(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_74(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_75(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_76(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_77(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_78(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_79(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_80(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_81(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_82(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_83(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_84(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_85(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_86(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_87(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_88(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_89(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_90(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_91(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_92(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_93(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_94(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_95(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_96(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_97(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_98(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_99(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_100(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_101(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_102(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_103(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_104(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_105(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_106(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_107(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_108(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_109(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_110(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_111(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_112(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_113(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_114(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_115(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_116(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_117(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_118(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_119(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_120(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_121(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_122(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_123(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_124(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_125(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_126(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_127(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_128(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_129(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_130(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_131(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_132(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_133(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_134(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_135(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_136(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_137(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_138(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_139(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_140(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_141(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_142(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_143(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_144(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_145(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_146(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_147(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_148(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_149(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_150(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_151(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_152(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_153(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_154(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_155(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_156(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_157(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_158(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_159(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_160(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_161(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_162(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_163(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_164(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_165(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_166(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_167(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_168(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_169(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_170(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_171(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_172(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_173(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_174(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_175(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_176(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_177(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_178(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_179(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_180(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_181(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_182(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_183(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_184(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_185(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_186(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_187(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_188(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_189(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_190(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_191(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_192(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_193(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_194(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_195(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_196(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_197(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_198(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_199(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_200(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_201(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_202(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_203(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_204(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_205(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_206(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_207(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_208(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_209(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_210(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_211(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_212(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_213(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_214(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_215(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_216(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_217(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_218(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_219(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_220(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_221(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_222(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_223(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_224(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_225(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_226(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_227(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_228(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_229(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_230(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_231(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_232(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_233(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_234(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_235(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_236(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_237(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_238(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_239(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_240(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_241(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_242(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_243(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_244(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_245(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_246(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_247(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_248(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_249(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

    def test_serving_case_250(self):
        cache = InferenceCache(capacity=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")
        cache.put("k4", "v4")
        self.assertIsNone(cache.get("k2"))

if __name__ == '__main__':
    unittest.main()
