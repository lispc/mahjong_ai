import algo
import tile


class Message:
    def __init__(self, sender, type, data):
        self.type = type
        self.data = data
        self.sender = sender


class Agent:
    def __init__(self, name, verbose=True):
        self.cur = []            # 闭手（不含副露）
        self.melds = []          # [(type, tile), ...] 已吃/碰/杠的副露
        self.name = name
        self.verbose = verbose

    def full_hand(self):
        """返回完整手牌：闭手 + 副露牌。"""
        return list(self.cur) + [t for _, t in self.melds]

    def add_meld(self, meld_type, tile_val):
        self.melds.append((meld_type, tile_val))

    def __eq__(self, other):
        return self.name == other.name

    def init_tiles(self, l):
        self.cur = l
        self.melds = []

    def handle_msg(self, msg):
        # 旧版本曾用 message 机制检测“这张弃牌能否胡”，现已由 respond_hu 统一处理，
        # 这里不再修改手牌，避免在副露/响应场景下手牌数错乱。
        from agent import Message
        return Message(self.name, 'no_op', None)

    def add(self, t):
        self.cur.append(t)
        ok = algo.is_succ(self.full_hand())
        if self.verbose:
            print('摸牌:' + tile.tile_to_str(t))
        #if ok:
        #    print(self.name + '自摸' + str(sorted(self.cur)))
        return ok

    def next(self):
        assert len(self.cur) >= 1
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

    # ---- 碰/杠/和响应接口（完整动作空间） ----
    def respond_hu(self, tile_val, context=None):
        """有人打出 tile_val，是否胡牌。基类用手牌判断。"""
        return algo.is_succ(self.full_hand() + [tile_val])

    def _can_peng(self, tile_val):
        return sum(1 for t in self.cur if t == tile_val) >= 2

    def _can_gang(self, tile_val):
        return sum(1 for t in self.cur if t == tile_val) >= 3

    def respond_peng(self, tile_val, context=None):
        """是否碰牌。基类默认不碰。"""
        return False

    def respond_gang(self, tile_val, context=None):
        """是否杠牌。基类默认不杠。"""
        return False

    def print(self):
        print(self.name + ':' + tile.display_tiles(self.cur))
