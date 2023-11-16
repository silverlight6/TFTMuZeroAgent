function isGameStateEmpty(gameState: GameState): boolean {
    if (!gameState) {
        return true;
    }

    return Object.keys(gameState.players).length === 0 || gameState.summaries.length === 0;
}

function boardLocationToIndex(location: number): [number, number] {
    const xLoc: number = Math.floor(location / 4);
    const yLoc: number = location - (xLoc * 4);

    // This is in the (7, 4) format that the sim uses.
    // We need to rotate it to the (4, 7) format that the board uses.
    // (0, 3) -> (0, 0), (0, 2) -> (1, 0) ...
    const x: number = 3 - yLoc;
    const y: number = xLoc;

    return [x, y]
}

function createEmptyBoard(): Board {
    const board: Board = [];
    for (let i = 0; i < 4; i++) {
        const row: (Champion | null)[] = [];
        for (let j = 0; j < 7; j++) {
            row.push(null);
        }
        board.push(row);
    }
    return board;
}

function createBoard(champions: Champion[]): Board {
    let board = createEmptyBoard()

    for (const champion of champions) {
        const [x, y] = boardLocationToIndex(champion.location);
        board[x][y] = champion;
    }
    return board;
}

function createEmptyBench(): Bench {
    return Array(9).fill(null);
}

function createBench(champions: Champion[]): Bench {
    let bench = createEmptyBench()
    for (const champion of champions) {
        bench[champion.location] = champion;
    }
    return bench

}

function createEmptyItemBench(): ItemBench {
    return Array(10).fill(null);
}

function createItemBench(items: Item[]): ItemBench {
    let itemBench = createEmptyItemBench()
    for (let i = 0; i < items.length; i++) {
        itemBench[i] = items[i];
    }
    return itemBench
}




function getCostColor(cost) {
    switch (cost) {
        case 1:
            return '#afafaf';
        case 2:
            return '#1bc660';
        case 3:
            return '#0b6cc3';
        case 4:
            return '#f947c6';
        case 5:
            return '#fe8900';
        default:
            return '#afafaf';
    }
}

function getCostBorder(cost) {
    return `bg-[${getCostColor(cost)}]`
}

function getStarColor(stars) {
    switch (stars) {
        case 1:
            return '#cd7f32';
        case 2:
            return '#C0C0C0';
        case 3:
            return '#FFD700';
        default:
            return '#FFD700';
    }
}

function getStars(stars) {
    if (!stars) {
        return [];
    }
    let color = getStarColor(stars);
    return Array(stars).fill(color);
}

export { isGameStateEmpty, boardLocationToIndex, createEmptyBoard, createBoard, createEmptyBench, createBench, createEmptyItemBench, createItemBench, getCostColor, getCostBorder, getStars };