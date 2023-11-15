interface PlayerState {
    health: number;
    exp: number;
    level: number;
    gold: number;
    win_streak: number;
    loss_streak: number;
    board: Champion[];
    bench: Champion[];
    shop: Champion[];
    items: Item[];
    traits: Trait[];
}

interface Champion {
    name: string;
    cost: number;
    stars: number;
    chosen: boolean | string;
    location: number;
    items: Item[];
}

interface Item {
    name: string;
}

interface Trait {
    name: string;
    count: number;
    total: number;
}

interface Action {
    action: number[];
    health: number | undefined;
    exp: number | undefined;
    level: number | undefined;
    gold: number | undefined;
    win_streak: number | undefined;
    loss_streak: number | undefined;
    board: Champion[] | undefined;
    bench: Champion[] | undefined;
    shop: Champion[] | undefined;
    items: Item[] | undefined;
    traits: Trait[] | undefined;
}

interface Battle {
    round: number;
    opponent: string;
    result: string;
    health: number;
    damage: number;
    board: Champion[] | null;
}

interface Player {
    state: PlayerState;
    actions: Action[];
    battles: Battle[];
}

interface Summary {
    player: string;
    placement: number;
    health: number;
    gold: number;
    level: number;
    board: Champion[];
}

interface PlayerDiff {
    health: boolean;
    exp: boolean;
    level: boolean;
    gold: boolean;
    win_streak: boolean;
    loss_streak: boolean;
    board: ChampionDiff[];
    bench: ChampionDiff[];
    shop: ChampionDiff[];
    items: ItemDiff[];
    itemChampionDiff: ItemDiff[];
}

interface ChampionDiff {
    location: number;
}

interface ItemDiff {
    name: string;
    index: number;
    location: number | undefined;
}

interface GameState {
    players: {
        [playerID: string]: Player;
    };
    summaries: Summary[];
}

interface UIState {
    players: {
        [playerID: string]: {
            states: PlayerState[],
            diffs: PlayerDiff[]
        }
    }
    summaries: Summary[];
}

export class Game {
    uiState: UIState;

    /**
     * Create a game state from a JSON object.
     * @param json JSON string representing a game state.
     */
    constructor(json: string) {
        const state: GameState = JSON.parse(json);
        this.uiState = this.createUIState(state);
    }

    createUIState(state: GameState): UIState {
        const uiState: UIState = {
            players: {},
            summaries: state.summaries
        };

        for (const playerID in state.players) {
            const player = state.players[playerID];
            const [states, diffs] = this.accumulate(player);
            uiState.players[playerID] = {
                states: states,
                diffs: diffs
            }
        }

        return uiState;
    }

    /**
     * Accumulate the initial player state and all actions into a list of player states.
     */
    accumulate(player: Player): [PlayerState[], PlayerDiff[]] {
        const states: PlayerState[] = [player.state];
        const diffs: PlayerDiff[] = [];

        player.actions.forEach(action => {
            const state = states[states.length - 1];
            // create new player state from previous state
            const newState = this.apply({ ...state }, action);
            const newDiff: PlayerDiff = this.diff(state, newState, action);
            states.push(newState);
            diffs.push(newDiff);
        })
        return [states, diffs];
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
                    diff.push({ location: champion.location });
                }
            } else {
                diff.push({ location: champion.location });
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
}
