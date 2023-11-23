type PlayerState = {
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

type Champion = {
    name: string;
    cost: number;
    stars: number;
    chosen: boolean | string;
    location: number;
    items: Item[];
}

type Item = {
    name: string;
}

type Trait = {
    name: string;
    count: number;
    total: number;
}

type Action = {
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

type Battle = {
    round: number;
    health: z.number();
    opponent: string;
    damage: number;
    opponentDamage: number
    board: Champion[];
    opponentBoard: Champion[] | null;
    result: string;
}

type Player = {
    state: PlayerState;
    actions: Action[];
    battles: Battle[];
}

type Summary = {
    player: string;
    placement: number;
    health: number;
    gold: number;
    level: number;
    board: Champion[];
}

type GameState = {
    players: {
        [playerID: string]: Player;
    };
    summaries: Summary[];
}

type PlayerDiff = {
    action: number[];
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

type ChampionDiff = {
    name: string;
    cost: number;
    stars: number;
    chosen: boolean | string;
    location: number;
}

type ItemDiff = {
    name: string;
    index: number;
    location: number | undefined;
}

type TimeStep = {
    index: number,
    round: number;
    action: number[];
    battle: Battle | null | undefined;
}

type UIState = {
    players: {
        [playerID: string]: {
            states: PlayerState[],
            diffs: PlayerDiff[],
            steps: TimeStep[]
        }
    }
    summaries: Summary[];
}

type Board = (Champion | null | undefined)[][];

type Bench = (Champion | null | undefined)[];

type Shop = (Champion | null | undefined)[];

type ItemBench = (Item | null | undefined)[];