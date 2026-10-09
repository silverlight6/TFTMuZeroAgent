"""Read-only Set 4 champion and trait projections from simulator definitions."""

from copy import deepcopy

from Simulator.battle import origin_class_stats, stats
from Simulator.game import pool_stats


BASE_STATS = ('AD', 'AS', 'HEALTH', 'ARMOR', 'MR', 'MANA', 'MAXMANA', 'RANGE')


def champion_summary(champion_id):
    return {'champion_id': champion_id, 'cost': stats.COST[champion_id],
            'traits': list(origin_class_stats.origin_class[champion_id])}


def search_champions(query='', cost=None, trait_id=None):
    return {'champions': [champion_summary(name) for name in sorted(stats.BASE_CHAMPION_LIST)
                         if query.lower() in name.lower()
                         and (cost is None or stats.COST[name] == cost)
                         and (trait_id is None or trait_id in origin_class_stats.origin_class[name])]}


def get_champion(champion_id):
    result = champion_summary(champion_id)
    bonus = next(({'stat': entry['stat'], 'value': entry['value']}
                  for entry in origin_class_stats.chosen if entry['champion'] == champion_id), None)
    result.update({
        'star_costs': [{'stars': stars, 'gold': pool_stats.cost_star_values[result['cost'] - 1][stars - 1]}
                       for stars in (1, 2, 3)],
        'base_stats': {name: deepcopy(getattr(stats, name)[champion_id]) for name in BASE_STATS},
        'rule_parameters': {name: deepcopy(table[champion_id]) for name, table in vars(stats).items()
                            if isinstance(table, dict) and champion_id in table
                            and name not in (*BASE_STATS, 'COST')},
        'special_attributes': {'chosen': {
            'eligible_traits': [trait for trait in result['traits'] if trait not in origin_class_stats.chosen_exclude],
            'bonus': bonus}, 'kayn_forms': ['kayn_shadowassassin', 'kayn_rhast'] if champion_id == 'kayn' else []},
        'description': None, 'ability_description': None,
        'unavailable_fields': ['description', 'ability_description'],
    })
    return result


def search_traits(query=''):
    return {'traits': [{'trait_id': name, 'thresholds': list(origin_class_stats.tiers[name])}
                       for name in sorted(origin_class_stats.tiers) if query.lower() in name.lower()]}


def get_trait_champions(trait_id):
    return {'trait_id': trait_id, 'champions': search_champions(trait_id=trait_id)['champions']}


def get_trait(trait_id):
    effects = {name: deepcopy(table[trait_id]) for name, table in vars(origin_class_stats).items()
               if isinstance(table, dict) and trait_id in table and name != 'tiers'}
    if trait_id == 'fortune':
        effects['fortune_returns'] = deepcopy(origin_class_stats.fortune_returns)
    return {'trait_id': trait_id, 'thresholds': list(origin_class_stats.tiers[trait_id]),
            'activation': 'exact' if trait_id == 'ninja' else 'minimum', 'effects': effects,
            'champion_ids': [entry['champion_id'] for entry in get_trait_champions(trait_id)['champions']],
            'chosen_eligible': trait_id not in origin_class_stats.chosen_exclude,
            'description': None, 'unavailable_fields': ['description']}
