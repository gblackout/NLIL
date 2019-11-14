from common import constants

PRED_DICT = {}
IND2PRED = {}


class Predicate:

    def __init__(self, name, var_types):
        """

        :param name:
            string
        :param var_types:
            list of strings
        """
        self.name = name
        self.var_types = var_types
        self.num_args = len(var_types)

    def __repr__(self):
        return '%s(%s)' % (self.name, ','.join(self.var_types))


class PredReg:

    def __init__(self):

        self.pred_dict = {}
        self.pred2ind = {}
        self.ind2pred = {}
        self.num_pred = 0

    def add(self, pred):
        assert type(pred) is Predicate

        self.pred_dict[pred.name] = pred
        sorted_name = sorted([pred_name for pred_name in self.pred_dict.keys()])
        self.ind2pred = dict([(ind, pred_name) for ind, pred_name in enumerate(sorted_name)])
        self.pred2ind = dict([(pred_name, ind) for ind, pred_name in enumerate(sorted_name)])
        self.num_pred += 1

    def get_numargs(self, pred):
        return self.get_class(pred).num_args

    def is_unp(self, pred):
        return self.get_numargs(pred) == 1

    def is_ident(self, pred):
        return self.get_class(pred).name == constants.IDENT_PHI

    def get_class(self, pred):
        if type(pred) is int:
            assert pred in self.ind2pred
            pred_class = self.pred_dict[self.ind2pred[pred]]
        elif type(pred) is str:
            assert pred in self.pred_dict
            pred_class = self.pred_dict[pred]
        elif type(pred) is Predicate:
            pred_class = pred
        else:
            raise ValueError

        return pred_class


pred_register = PredReg()