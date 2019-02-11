'''
a simple tree structure to help get shift-reduce actions
'''
class tree(object):

    def __init__(self, tokens, arcs):
        self.nodes = {token: node(name=token) for token in tokens}
        for arc in arcs:
            i, j = arc
            self.nodes[j].set_parent(self.nodes[i])

    def remove_node(self, idx):
        if len(self.nodes[idx].children) > 0 or not self.nodes[idx].parent:
            return False
        else:
            self.nodes[idx].parent.children.remove(self.nodes[idx])
            del self.nodes[idx]
            return True

    def get_nodes(self):
        return self.nodes

    def show(self):
        for name in self.nodes:
            print self.nodes[name], '->', self.nodes[name].children

    def _depth(self, n):
        if len(n.children) == 0:
            return 1
        else:
            return max([1 + self._depth(child) for child in n.children])

    def get_depth(self):
        root = None
        for token in self.nodes:
            if not self.nodes[token].parent:
                root = self.nodes[token]
                break
        if not root:
            raise ValueError('no root node!')
        return self._depth(root)


class node(object):
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent
        self.parent.children.append(self)

    def get_name(self):
        return self.name

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    tokens = range(1, 6)
    arcs = [(2, 1), (2, 4), (4, 3), (4, 5)]

    t = tree(tokens, arcs)
    t.show()
    print 'tree depth', t.get_depth()

    print '*' * 10
    t.remove_node(3)
    t.show()

    print '*' * 10
    t.remove_node(5)
    t.show()

    print '*' * 10
    t.remove_node(4)
    t.show()

    print '*' * 10
    t.remove_node(1)
    t.show()

    print '*' * 10
    t.remove_node(2)
    t.show()

else:
    pass
