import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
import math
global posWalls, posGoals


class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""

    def __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0


"""Load puzzles and define the rules of sokoban"""


def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n', '') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ':
                layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#':
                layout[irow][icol] = 1  # wall
            elif layout[irow][icol] == '&':
                layout[irow][icol] = 2  # player
            elif layout[irow][icol] == 'B':
                layout[irow][icol] = 3  # box
            elif layout[irow][icol] == '.':
                layout[irow][icol] = 4  # goal
            elif layout[irow][icol] == 'X':
                layout[irow][icol] = 5  # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum - colsNum)])

    # print(layout)
    return np.array(layout)


def transferToGameState2(layout, player_pos):
    maxColsNum = max(len(x) for x in layout)

    temp = np.empty((len(layout), maxColsNum), dtype=object)

    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = val

    return temp


def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0])  # e.g. (2, 2)


def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (
        gameState == 5)))  # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))


def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(
        gameState == 1))  # e.g. like those above


def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(
        tuple(x) for x in np.argwhere(
            (gameState == 4) | (
                gameState == 5)))  # e.g. like those above


def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)


def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper():  # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls


def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1, 0, 'u', 'U'], [1, 0, 'd', 'D'],
                  [0, -1, 'l', 'L'], [0, 1, 'r', 'R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox:  # the move was a push
            action.pop(2)  # drop the little letter
        else:
            action.pop(3)  # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else:
            continue

    # e.g. ((0, -1, 'l'), (0, 1, 'R'))
    return tuple(tuple(x) for x in legalActions)


def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer  # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer +
                    action[1]]  # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper():  # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox


def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8][::-1],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6][::-1]]
    flipPattern = [[2, 1, 0, 5, 4, 3, 8, 7, 6],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8],
                   [2, 1, 0, 5, 4, 3, 8, 7, 6][::-1],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] -
                      1, box[1] -
                      1), (box[0] -
                     1, box[1]), (box[0] -
                                  1, box[1] +
                                  1), (box[0], box[1] -
                     1), (box[0], box[1]), (box[0], box[1] +
                                            1), (box[0] +
                                                 1, box[1] -
                                                 1), (box[0] +
                     1, box[1]), (box[0] +
                                  1, box[1] +
                                  1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox:
                    return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls:
                    return True
    return False


"""Implement all approcahes"""


def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]]
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp


def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    # Lay vi tri ban dau cua cac hop trong sokoban
    beginBox = PosOfBoxes(gameState)
    # lay vi tri ban dau cua nhan vat
    beginPlayer = PosOfPlayer(gameState)
    # luu vi tri bat dau cua cac hop va nhan vat
    startState = (beginPlayer, beginBox)
    # frontier luu cac duong di trang thai can kham pha (queue cua BFS)
    frontier = collections.deque([[startState]])
    # luu cac trang thai da duyet de tranh duyet lap lai
    exploredSet = set()
    # luu cac hanh dong tuong ung voi trang thai trong frontier
    actions = collections.deque([[0]])
    # luu ket qua tu dau den cuoi
    temp = []
    ### CODING FROM HERE ###
    # lap lai den khi frontier rong
    while frontier:

        # lay va xoa phan tu dau tien cua frontier, (phan tu trai nhat cua day)
        node = frontier.popleft()

        # lay chuoi hanh dong tuong ung voi node
        node_action = actions.popleft()

        # kiem tra xem trang thai hien cac hop toi vi tri goal hay chua
        if isEndState(node[-1][-1]):

            # them cac action cua trang thai hien tai vao temp
            temp += node_action[1:]
            # thoat lap
            break

        # neu trang thai chua duoc duyet
        if node[-1] not in exploredSet:

            # them trang thai vao da duyet
            exploredSet.add(node[-1])

            # lay cac hanh dong hop le cua nhan vat ung voi trang thai hien tai
            for action in legalActions(node[-1][0], node[-1][1]):

                # tinh trang thai moi khi thuc hien hanh duong
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)

                # kiem tra box von co the di chuyen khong
                if isFailed(newPosBox):
                    continue

                # them  trang thai moi vao frontier
                frontier.append(node + [(newPosPlayer, newPosBox)])

                # them cac hanh dong moi vao action
                actions.append(node_action + [action[-1]])

    # tra ve danh sach hanh dong tu dau den cuoi
    return temp


def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])


def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    start = time.time()
    # Lay vi tri ban dau cua cac hop trong sokoban
    beginBox = PosOfBoxes(gameState)
    # lay vi tri ban dau cua nhan vat
    beginPlayer = PosOfPlayer(gameState)
    # luu vi tri bat dau cua cac hop va nhan vat
    startState = (beginPlayer, beginBox)
    # frontier de luon lay node co cost nho nhat
    frontier = PriorityQueue()
    # them trang thai ban dau co cost = 0
    frontier.push([startState], 0)
    # luu cac trang thai da duyet de tranh lap lai
    exploredSet = set()
    # luu chuoi hanh dong tuong ung moi node trong frontier
    actions = PriorityQueue()
    # them hanh dong bat dau
    actions.push([0], 0)
    # danh sach luu ket qua cuoi cung
    temp = []
    ### CODING FROM HERE ###
    # thuc hien hanh dong khi frontier co phan tu
    while not frontier.isEmpty():

        # lay va xoa trang thai cost nho nhat
        node = frontier.pop()

        # lay va xoa hanh dong tuong ung co trang thai do tu action
        node_action = actions.pop()

        # kiem tra xem trang thai hien cac hop toi vi tri goal hay chua
        if isEndState(node[-1][-1]):

            # them cac hanh dong cua trang thai hien tai vao temp
            temp += node_action[1:]

            # thoat lap
            break

        # neu trang thai chua duoc duyet
        if node[-1] not in exploredSet:

            # them trang thai vao explored
            exploredSet.add(node[-1])

            # duyet cac hanh dong hop le ung voi trang thai hien tai
            for action in legalActions(node[-1][0], node[-1][1]):

                # update trang thai moi sau khi hanh dong
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)

                # kiem tra trang thai cua box khong the di chuyen hay khong
                if isFailed(newPosBox):
                    # bo qua hanh dong hien tai
                    continue

                # them trang thai moi vao frontier va chi phi duong di
                frontier.push(
                    node + [(newPosPlayer, newPosBox)], cost(node_action[1:]))

                # them chuoi hanh dong tuong ung va chi phi
                actions.push(node_action + [action[-1]], cost(node_action[1:]))
    runtime = time.time() - start
    print(f"Runtime of UCS: {runtime:.2f} s")
    # In số lượng nút đã được mở ra
    print('Explored Nodes: ', len(exploredSet))
    # trả về các hành động từ đầu đến cuối
    return temp


def heuristic_manhattan(posPlayer, posBox):
    # khởi tạo biến để lưu tổng khoảng cách manhattan
    distance = 0

    # lấy các box đã nằm đúng goal
    completes = set(posGoals) & set(posBox)

    # lấy các box chưa tới goal
    remainBox = list(set(posBox).difference(completes))

    # lấy các goal chưa có box
    remainGoal = list(set(posGoals).difference(completes))

    # tính Manhattan distance
    for i in range(len(remainBox)):
        # tính khoảng cách Manhattan giữa box và goal tương ứng
        # manhattan distance = |x1 - x2| + |y1 - y2|
        distance += abs(remainBox[i][0] - remainGoal[i][0]) + \
            abs(remainBox[i][1] - remainGoal[i][1])

    return distance


def aStarSearch_manhattan(gameState):
    start = time.time()
    """A* search using Manhattan heuristic"""
    # lấy vị trí ban đầu của các box
    beginBox = PosOfBoxes(gameState)
    # lấy vị trí ban đầu của player
    beginPlayer = PosOfPlayer(gameState)
    # danh sách lưu kết quả cuối cùng
    temp = []
    # tạo trạng thái bắt đầu (player, box)
    start_state = (beginPlayer, beginBox)
    # frontier lưu các trạng thái cần được mở rộng
    frontier = PriorityQueue()
    # priority = f(n) = g(n) + h(n)
    # ở bước đầu g(n) = 0 nên priority = h(n)
    frontier.push([start_state], heuristic_manhattan(beginPlayer, beginBox))
    # tập lưu các trạng thái đã được khám phá
    exploredSet = set()
    # PriorityQueue lưu chuỗi hành động tương ứng với mỗi trạng thái
    actions = PriorityQueue()
    # thêm hành động ban đầu vào actions
    actions.push([0], heuristic_manhattan(beginPlayer, beginBox))

    # lặp khi frontier chưa rỗng
    while not frontier.isEmpty():

        # lấy node có f(n) nhỏ nhất từ frontier
        node = frontier.pop()
        # lấy chuỗi hành động tương ứng với node đó
        node_action = actions.pop()

        # kiểm tra nếu tất cả các box đã tới goal
        if isEndState(node[-1][-1]):
            # lưu chuỗi hành động
            temp += node_action[1:]
            # kết thúc thuật toán
            break

        # nếu chưa duyệt
        if node[-1] not in exploredSet:
            # thêm vào tập đã duyệt
            exploredSet.add(node[-1])

            # duyệt tất cả các hành động hợp lệ của player
            for action in legalActions(node[-1][0], node[-1][1]):
                # cập nhật trạng thái mới sau khi thực hiện hành động
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                # nếu không di chuyển được thì skip
                if isFailed(newPosBox):
                    continue
                # tính chi phí g(n)
                # g(n) = số bước di chuyển của player
                g = cost(node_action[1:])

                # tính heuristic h(n)
                # h(n) = khoảng cách Manhattan từ box tới goal
                h = heuristic_manhattan(newPosPlayer, newPosBox)

                # priority = f(n) = g(n) + h(n)
                frontier.push(node + [(newPosPlayer, newPosBox)], g + h)
                # lưu chuỗi hành động tương ứng
                actions.push(node_action + [action[-1]], g + h)
    runtime = time.time() - start
    print(f"Runtime of A* manhattan: {runtime:.2f} s")
    # In số lượng nút đã được mở ra
    print('Explored Nodes: ', len(exploredSet))
    # trả về các hành động từ đầu đến cuối
    return temp


def heuristic_euclidean(posPlayer, posBox):
    # khởi tạo biến distance để lưu tổng khoảng cách Euclidean
    distance = 0
    # lấy các box đã nằm đúng goal
    completes = set(posGoals) & set(posBox)
    # lấy các box chưa tới goal
    remainBox = list(set(posBox).difference(completes))
    # lấy các goal chưa có box
    remainGoal = list(set(posGoals).difference(completes))
    # tính khoảng cách Euclidean giữa box và goal tương ứng
    # sqrt((x1-x2)^2 + (y1-y2)^2)
    for i in range(len(remainBox)):
        distance += math.sqrt((remainBox[i][0] -
                               remainGoal[i][0])**2 +
                              (remainBox[i][1] -
                               remainGoal[i][1])**2)
    # trả về tổng khoảng cách Euclidean
    return distance


def aStarSearch_euclidean(gameState):
    start = time.time()
    """A* search using Euclidean heuristic"""
    # lấy vị trí ban đầu của các box trong game
    beginBox = PosOfBoxes(gameState)
    # lấy vị trí ban đầu của player
    beginPlayer = PosOfPlayer(gameState)
    # tạo trạng thái bắt đầu
    start_state = (beginPlayer, beginBox)
    # frontier dùng để lưu các trạng thái cần được mở rộng
    frontier = PriorityQueue()
    # thêm trạng thái ban đầu vào frontier
    frontier.push([start_state], heuristic_euclidean(beginPlayer, beginBox))
    # tập hợp lưu các trạng thái đã được duyệt
    exploredSet = set()
    # lưu chuỗi hành động tương ứng với mỗi trạng thái
    actions = PriorityQueue()
    # thêm hành động ban đầu vào actions
    actions.push([0], heuristic_euclidean(beginPlayer, beginBox))
    # danh sách lưu kết quả cuối cùng
    temp = []
    # lặp khi frontier vẫn còn trạng thái cần duyệt
    while not frontier.isEmpty():
        # lấy trạng thái có f(n) nhỏ nhất từ frontier
        node = frontier.pop()
        # lấy chuỗi hành động tương ứng với trạng thái đó
        node_action = actions.pop()
        # kiểm tra nếu tất cả các box đã tới goal
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            # kết thúc thuật toán
            break
        # nếu trạng thái hiện tại chưa được duyệt
        if node[-1] not in exploredSet:
            # thêm trạng thái vào tập explored
            exploredSet.add(node[-1])
            # duyệt qua tất cả các hành động hợp lệ
            for action in legalActions(node[-1][0], node[-1][1]):
                # cập nhật trạng thái mới sau khi thực hiện hành động
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                # nếu hộp bị mắc kẹt thì bỏ qua
                if isFailed(newPosBox):
                    continue
                # g(n) = số bước di chuyển của player
                g = cost(node_action[1:])
                # h(n) = khoảng cách Euclidean từ box tới goal
                h = heuristic_euclidean(newPosPlayer, newPosBox)
                # priority = f(n) = g(n) + h(n)
                frontier.push(node + [(newPosPlayer, newPosBox)], g + h)
                # lưu chuỗi hành động tương ứng
                actions.push(node_action + [action[-1]], g + h)
    runtime = time.time() - start
    print(f"Runtime of A* euclidean: {runtime:.2f} s")

    # In số lượng nút đã được mở ra
    print('Explored Nodes: ', len(exploredSet))
    # Trả về danh sách hành động
    return temp


"""Read command"""


def readCommand(argv):
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels, "r") as f:
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args


def get_move(layout, player_pos, method):
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)

    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'astar_manhattan':
        result = aStarSearch_manhattan(gameState)
    elif method == 'astar_euclidean':
        result = aStarSearch_euclidean(gameState)
    else:
        raise ValueError('Invalid method.')
    if len(result) == 0:
        print("No solution found")
    print('Total: ', len(result), ' steps')
    print(result)
    return result
