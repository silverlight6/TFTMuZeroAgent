export function locationTo2DIndex(location: number): [number, number] {
    const xLoc: number = Math.floor(location / 4);
    const yLoc: number = location - xLoc * 4;

    // This is in the (7, 4) format that the sim uses.
    // We need to rotate it to the (4, 7) format that the board uses.
    // (0, 3) -> (0, 0), (0, 2) -> (1, 0) ...
    const x: number = 3 - yLoc;
    const y: number = xLoc;

    return [x, y];
}

export function computeRoundString(round: number): string {
    if (round < 3) {
        return `1 - ${round + 1}`
    } else {
        round = round - 3;
        let roundNum = Math.floor(round / 6) + 2;
        let roundStage = (round % 6) + 1;
        return `${roundNum} - ${roundStage}`;
    }
}