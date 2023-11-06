import numpy as np

"""
Description - Code taken from https://www.geeksforgeeks.org/max-heap-in-python/ and adjusted for the use-case
Inputs      -
    maxsize
        The maximum size of the priority queue.
"""
class PriorityBuffer(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [0] * (self.maxsize + 1)
        self.Replay_Heap = [None] * (self.maxsize + 1)
        self.Heap[0] = 0
        self.FRONT = 1

    def qsize(self):
        return self.size

    '''
    Description - 
        Function to return the position of parent for the node currently at pos
    '''
    def parent(self, pos):
        return pos // 2

    '''
    Description -
        Function to return the position of the left child for the node currently at pos
    '''
    def leftChild(self, pos):
        return 2 * pos

    '''
    Description - 
        Function to return the position of the right child for the node currently at pos
    '''
    def rightChild(self, pos):
        return (2 * pos) + 1

    '''
    Description -
        Function that returns true if the passed node is a leaf node
    '''
    def isLeaf(self, pos):
        if (self.size // 2) <= pos <= self.size:
            return True
        return False

    '''
    Description - 
        Function to swap two nodes of the heap
    '''
    def swap(self, fpos, spos):
        self.Heap[fpos], self.Heap[spos] = (self.Heap[spos], self.Heap[fpos])
        self.Replay_Heap[fpos], self.Replay_Heap[spos] = (self.Replay_Heap[spos], self.Replay_Heap[fpos])

    '''
    Description - 
        Function to heapify the node at pos
    '''
    def maxHeapify(self, pos):
        # If the node is a non-leaf node and smaller
        # than any of its child
        if not self.isLeaf(pos):
            if (self.Heap[pos] < self.Heap[self.leftChild(pos)] or
                    self.Heap[pos] < self.Heap[self.rightChild(pos)]):

                # Swap with the left child and heapify
                # the left child
                if (self.Heap[self.leftChild(pos)] >
                        self.Heap[self.rightChild(pos)]):
                    self.swap(pos, self.leftChild(pos))
                    self.maxHeapify(self.leftChild(pos))

                # Swap with the right child and heapify
                # the right child
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.maxHeapify(self.rightChild(pos))

    '''
    Description - 
        Function to insert a node into the heap
    '''
    def insert(self, priority, time_step):
        # If max size, check a random place and see if the priority is higher.
        if self.size >= self.maxsize:
            # Start of the leaf positions to the end of the array
            random_pos = np.random.randint(self.maxsize // 2 + 1, self.maxsize - 1)
            if priority > self.Heap[random_pos]:
                self.Heap[random_pos] = priority
                self.Replay_Heap[random_pos] = time_step
                while (self.Heap[random_pos] >
                       self.Heap[self.parent(random_pos)]):
                    self.swap(random_pos, self.parent(random_pos))
                    random_pos = self.parent(random_pos)
            return
        self.size += 1
        self.Heap[self.size] = priority
        self.Replay_Heap[self.size] = time_step

        current = self.size

        while (self.Heap[current] >
               self.Heap[self.parent(current)]):
            self.swap(current, self.parent(current))
            current = self.parent(current)

    '''
    Description - 
        Function to print the contents of the heap. Here for debugging purposes.
    '''
    def Print(self):
        for i in range(1, (self.size // 2) + 1):
            print("PARENT : " + str(self.Heap[i]) +
                  "LEFT CHILD : " + str(self.Heap[2 * i]) +
                  "RIGHT CHILD : " + str(self.Heap[2 * i + 1]))

    '''
    Description - 
        Function to remove and return the maximum element from the heap
    '''
    def extractMax(self):
        popped = self.Replay_Heap[self.FRONT]
        priority_popped = self.Heap[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.Replay_Heap[self.FRONT] = self.Replay_Heap[self.size]
        self.size -= 1
        self.maxHeapify(self.FRONT)
        return popped, priority_popped
