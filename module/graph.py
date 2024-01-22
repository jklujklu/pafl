def harary(n: int, k: int):
    r = int(k / 2)
    graph = []
    nodes = set()
    neighbors = {}
    for x in range(0, n):
        nodes.add(x)
        neighbors[x] = set()

    for x in range(0, n):
        for y in range(1, r + 1):
            graph.append((x, (x + y) % n))

            neighbors[(x + y) % n].add(x)
            neighbors[x].add((x + y) % n)
    if (n % 2 == 0) and (k % 2 != 0):
        for x in range(0, int(n / 2)):
            graph.append((x, x + int(n / 2)))

            neighbors[x + int(n / 2)].add(x)
            neighbors[x].add(x + int(n / 2))
    if (n % 2 != 0) and (k % 2 != 0):
        for x in range(0, int(n / 2) + 1):
            graph.append((x, x + int(n / 2)))

            neighbors[(x + int(n / 2))].add(x)
            neighbors[x].add((x + int(n / 2)))
        graph.append((0, int((n - 1) / 2)))

        neighbors[int((n - 1) / 2)].add(0)
        neighbors[0].add(int((n - 1) / 2))

    return graph, list(nodes), neighbors


if __name__ == '__main__':
    n = int(input("Enter value of n:"))
    k = int(input("Enter value of k:"))

    if k > n - 1:
        print("Graph with k greater than or equal to n is not possible\n")
    else:
        harary(n, k)
