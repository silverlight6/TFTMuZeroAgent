import { z } from 'zod';
import { toZod } from 'tozod';

export const Item: toZod<Item> = z.object({
    name: z.string(),
});

export const Champion: toZod<Champion> = z.object({
    name: z.string(),
    cost: z.number(),
    stars: z.number(),
    chosen: z.union([z.boolean(), z.string()]),
    location: z.number(),
    items: z.array(Item),
});

export const Trait: toZod<Trait> = z.object({
    name: z.string(),
    count: z.number(),
    total: z.number(),
});

export const PlayerState: toZod<PlayerState> = z.object({
    health: z.number(),
    exp: z.number(),
    level: z.number(),
    gold: z.number(),
    win_streak: z.number(),
    loss_streak: z.number(),
    board: z.array(Champion),
    bench: z.array(Champion),
    shop: z.array(Champion),
    items: z.array(Item),
    traits: z.array(Trait),
});

export const Action: toZod<Action> = z.object({
    action: z.array(z.number()),
    health: z.number().optional(),
    exp: z.number().optional(),
    level: z.number().optional(),
    gold: z.number().optional(),
    win_streak: z.number().optional(),
    loss_streak: z.number().optional(),
    board: z.array(Champion).optional(),
    bench: z.array(Champion).optional(),
    shop: z.array(Champion).optional(),
    items: z.array(Item).optional(),
    traits: z.array(Trait).optional(),
});

export const Battle: toZod<Battle> = z.object({
    round: z.number(),
    health: z.number(),
    opponent: z.string(),
    damage: z.number(),
    opponentDamage: z.number(),
    board: z.array(Champion),
    opponentBoard: z.array(Champion).nullable(),
    result: z.string(),
});


export const Player: toZod<Player> = z.object({
    state: PlayerState,
    actions: z.array(Action),
    battles: z.array(Battle),
});

export const Summary: toZod<Summary> = z.object({
    player: z.string(),
    placement: z.number(),
    health: z.number(),
    gold: z.number(),
    level: z.number(),
    board: z.array(Champion),
});

export const GameState: toZod<GameState> = z.object({
    players: z.record(Player),
    summaries: z.array(Summary),
});