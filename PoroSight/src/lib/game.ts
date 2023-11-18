
export class Game {
    uiState: UIState;

    /**
     * Create a game state from a JSON object.
     * @param json JSON string representing a game state.
     */
    constructor(state: GameState) {
        this.uiState = this.createUIState(state);
        this.uiState.summaries.sort((a, b) => a.placement - b.placement);
    }

    getPlayers(): string[] {
        return this.uiState.summaries.map(summary => summary.player);
    }

    getPlayerLength(playerID: string): number {
        return Math.min(this.uiState.players[playerID].states.length, this.uiState.players[playerID].diffs.length) - 1;
    }

    getPlayerState(playerID: string, index: number): [PlayerState, PlayerDiff] {
        return [this.uiState.players[playerID].states[index], this.uiState.players[playerID].diffs[index]];
    }

    getPlayerSummary(playerID: string): Summary {
        return this.uiState.summaries.filter(summary => summary.player === playerID)[0];
    }

    getPlayerSteps(playerID: string): TimeStep[] {
        return this.uiState.players[playerID].steps;
    }

    createUIState(state: GameState): UIState {
        const uiState: UIState = {
            players: {},
            summaries: []
        };

        if (!state) {
            return uiState;
        }

        for (const playerID in state.players) {
            const player = state.players[playerID];
            const [states, diffs, steps] = this.accumulate(player);
            uiState.players[playerID] = {
                states: states,
                diffs: diffs,
                steps: steps,
            }
        }

        if (state.summaries) {
            uiState.summaries = state.summaries;
        }

        return uiState;
    }

    /**
     * Accumulate the initial player state and all actions into a list of player states.
     */
    accumulate(player: Player): [PlayerState[], PlayerDiff[], TimeStep[]] {
        const states: PlayerState[] = [player.state];
        const diffs: PlayerDiff[] = [this.emptyDiff()];
        const steps: TimeStep[] = [{
            index: 0,
            round: 0,
            action: [-2, -2, -2],
            battle: player.battles[0]
        }]
        let round = 1;

        player.actions.forEach((action, index) => {
            const state = states[states.length - 1];
            // create new player state from previous state
            const newState = this.apply({ ...state }, action);
            const newDiff: PlayerDiff = this.diff(state, newState, action);
            let timeStep: TimeStep;
            if (this.isBattle(action)) {
                timeStep = this.step(index, round, action, player.battles[round]);
                round += 1;
            } else {
                timeStep = this.step(index, round, action);
            }
            states.push(newState);
            diffs.push(newDiff);
            steps.push(timeStep);
        })
        return [states, diffs, steps];
    }

    /**
     * Apply an action to a player state.
     */
    apply(state: PlayerState, action: Action): PlayerState {
        Object.keys(state).forEach(key => {
            const typedKey = key as keyof PlayerState;
            if (action[typedKey] !== null && action[typedKey] !== undefined) {
                state[typedKey] = action[typedKey];
            }
        });

        return state;
    }

    /**
     * Calculate the difference between two player states.
     */
    diff(state: PlayerState, newState: PlayerState, action: Action): PlayerDiff {
        const newDiff: PlayerDiff = {
            action: action.action,
            health: false,
            exp: false,
            level: false,
            gold: false,
            win_streak: false,
            loss_streak: false,
            board: [],
            bench: [],
            shop: [],
            items: [],
            itemChampionDiff: [],
        };

        // Check if key exists in action
        const keyExists = (key: string) => {
            return action[key] !== null && action[key] !== undefined;
        }

        if (keyExists("health")) {
            newDiff.health = state.health !== newState.health;
        }
        if (keyExists("exp")) {
            newDiff.exp = state.exp !== newState.exp;
        }
        if (keyExists("level")) {
            newDiff.level = state.level !== newState.level;
        }
        if (keyExists("gold")) {
            newDiff.gold = state.gold !== newState.gold;
        }
        if (keyExists("win_streak")) {
            newDiff.win_streak = state.win_streak !== newState.win_streak;
        }
        if (keyExists("loss_streak")) {
            newDiff.loss_streak = state.loss_streak !== newState.loss_streak;
        }
        if (keyExists("items")) {
            newDiff.items = this.itemBenchDiff(state.items, newState.items);
        }

        const [action_type, from, to] = action.action;
        // Refresh Action
        if (action_type == 2 && keyExists("shop")) {
            newDiff.shop = this.championDiff(state.shop, newState.shop);
        }
        // Buy Action
        else if (action_type == 3) {
            if (keyExists("shop")) {
                newDiff.shop = this.championDiff(state.shop, newState.shop);
            }
            if (keyExists("bench")) {
                newDiff.bench = this.championDiff(state.bench, newState.bench);
            }
            if (keyExists("board")) {
                newDiff.board = this.championDiff(state.board, newState.board);
            }
        }

        // Sell Action
        else if (action_type == 4) {
            if (from < 28 && keyExists("board")) {
                newDiff.board = this.championDiff(state.board, newState.board);
            } else if (keyExists("bench")) {
                newDiff.bench = this.championDiff(state.bench, newState.bench);
            }
        }

        // Move Action
        else if (action_type == 5) {
            if (from < 28 && to < 28) { // Board to Board
                newDiff.board = this.championDiff(state.board, newState.board);
            } else if (from < 28 && to >= 28) { // Board to Bench
                newDiff.board = this.championDiff(state.board, newState.board);
                newDiff.bench = this.championDiff(state.bench, newState.bench);
            } else if (from >= 28 && to < 28) { // Bench to Board
                newDiff.bench = this.championDiff(state.bench, newState.bench);
                newDiff.board = this.championDiff(state.board, newState.board);
            } else { // Bench to Bench
                newDiff.bench = this.championDiff(state.bench, newState.bench);
            }
        }

        // Item Action
        else if (action_type == 6) {
            newDiff.itemChampionDiff = this.itemChampionDiff(state.items, newState.items, to);
        }

        // Special Action for Battles
        // Update board, bench, shop, items
        else if (action_type == -1) {
            if (keyExists("board")) {
                newDiff.board = this.championDiff(state.board, newState.board);
            }
            if (keyExists("health")) {
                newDiff.health = state.health !== newState.health;
            }
            if (keyExists("items")) {
                newDiff.items = this.itemBenchDiff(state.items, newState.items);
            }
            if (keyExists("shop")) {
                newDiff.shop = this.championDiff(state.shop, newState.shop);
            }
        }

        return newDiff;
    }


    /**
     * Calculate the difference between two lists of champions
     */
    championDiff(champions: Champion[], newChampions: Champion[]): ChampionDiff[] {
        // Create a location map for the old champions
        const locationMap: { [location: number]: Champion } = {};

        champions.forEach(champion => {
            locationMap[champion.location] = champion;
        });

        const diff: ChampionDiff[] = [];

        newChampions.forEach(champion => {
            const oldChampion = locationMap[champion.location];
            if (oldChampion) {
                if (oldChampion.name !== champion.name || oldChampion.stars !== champion.stars) {
                    diff.push({ ...champion });
                }
            } else {
                diff.push({ ...champion });
            }
        });

        return diff;
    }

    /**
     * Calculate the difference between two item benches
     */
    itemBenchDiff(items: Item[], newItems: Item[]): ItemDiff[] {
        const diff: ItemDiff[] = [];

        newItems.forEach((item, index) => {
            if (items[index] && items[index].name !== item.name) {
                diff.push({ name: item.name, index: index, location: undefined });
            }
        });

        return diff;
    }

    /**
     * Calculate the difference between two lists of items on champions
     */
    itemChampionDiff(items: Item[], newItems: Item[], location: number): ItemDiff[] {
        const diff: ItemDiff[] = [];

        newItems.forEach((item, index) => {
            if (items[index]) {
                if (items[index].name !== item.name) {
                    diff.push({ name: item.name, index: index, location: location });
                }
            } else {
                diff.push({ name: item.name, index: index, location: location });
            }
        })

        return diff;
    }

    /**
     * Empty PlayerDiff
     */
    emptyDiff(): PlayerDiff {
        return {
            action: [0, 0, 0],
            health: false,
            exp: false,
            level: false,
            gold: false,
            win_streak: false,
            loss_streak: false,
            board: [],
            bench: [],
            shop: [],
            items: [],
            itemChampionDiff: [],
        };
    }

    /**
     * Time Step
     */
    step(index: number, round: number, action: Action, battle?: Battle): TimeStep {
        return {
            index: index + 1,
            round: round,
            action: action.action,
            battle: battle
        }
    }

    isBattle(action: Action) {
        return action.action[0] === -2;
    }
}
