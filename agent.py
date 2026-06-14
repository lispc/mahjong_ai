import algo
import tile


class Message:
    def __init__(self, sender, type, data):
        self.type = type
        self.data = data
        self.sender = sender


class Agent:
    def __init__(self, name, verbose=True):
        self.cur = []
        self.name = name
        self.verbose = verbose

    def __eq__(self, other):
        return self.name == other.name

    def init_tiles(self, l):
        self.cur = l

    def handle_msg(self, msg):
        if msg.type == 'put' and msg.sender != self.name:
            t = msg.data
            if algo.is_succ(self.cur + [t]):
                self.cur.append(t)
                return Message(self.name, 'i_win', None)
        return Message(self.name, 'no_op', None)

    def add(self, t):
        self.cur.append(t)
        ok = algo.is_succ(self.cur)
        if self.verbose:
            print('摸牌:' + tile.tile_to_str(t))
        #if ok:
        #    print(self.name + '自摸' + str(sorted(self.cur)))
        return ok

    def next(self):
        assert len(self.cur) == 14
        result = algo.select(self.cur, False)[0]
        self.cur.remove(result)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result

    def declare_tenpai(self, hand, context):
        """
        返回是否报听。hand 为弃牌后的 13 张手牌。
        基类默认不报听；子类可覆盖。
        """
        return False

    def print(self):
        print(self.name + ':' + tile.display_tiles(self.cur))