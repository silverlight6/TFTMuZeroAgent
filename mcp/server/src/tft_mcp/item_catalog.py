"""Read-only Set 4 item definitions and source-supported assignment constraints."""

from copy import deepcopy

from Simulator.battle import item_stats


CONSUMABLES = {'kayn_rhast', 'kayn_shadowassassin', 'champion_duplicator',
               'magnetic_remover', 'reforger'}


def item_summary(item_id):
    kind = 'component' if item_id in item_stats.basic_items else (
        'consumable' if item_id in CONSUMABLES else 'equipment')
    return {'item_id': item_id, 'kind': kind, 'craftable': item_id in item_stats.item_builds}


def search_items(query='', kind=None):
    return {'items': [item_summary(item_id) for item_id in sorted(item_stats.items)
                      if query.lower() in item_id.lower()
                      and (kind is None or item_summary(item_id)['kind'] == kind)]}


def get_item(item_id):
    summary = item_summary(item_id)
    granted_trait = next((trait for trait, item in item_stats.trait_items.items()
                          if item == item_id), None)
    constraints = ['Requires a present unit that is not a target dummy.']
    if item_id in {'kayn_rhast', 'kayn_shadowassassin'}:
        constraints += [
            'Targets Kayn only and consumes both Kayn form items.',
            'Form items become available after three Kayn board rounds with two inventory vacancies.',
            'Source limitation: stored form IDs differ from combat ability checks; bench assignment writes kaynform instead of kayn_form. Combat transformation is not guaranteed.']
    elif item_id == 'champion_duplicator':
        constraints += ['Requires nonzero champion cost and a bench vacancy.',
                        'Creates a new default-star champion preserving chosen and form arguments; consumes the duplicator.']
    elif item_id in {'magnetic_remover', 'reforger'}:
        constraints += ['Requires equipped items and inventory room for their full count before this consumable is removed.']
        if item_id == 'magnetic_remover':
            constraints += ['Returns equipped items to inventory and consumes the remover.']
        else:
            constraints += ['Returns random replacements from source item categories and consumes the reforger; spatula remains spatula.']
    else:
        constraints += ['Maximum three equipped items.',
                        'Two components may combine at the three-item limit when the final equipped item is a component.']
        if granted_trait:
            constraints += ['Rejects a unit already possessing the granted trait.']
        if item_id == 'thieves_gloves':
            constraints += ['Requires no equipped items; occupies equipment with thieves_gloves and two distinct random items from thieves_gloves_items.']
    effects = {name: deepcopy(table[item_id]) for name, table in vars(item_stats).items()
               if not name.startswith("_") and isinstance(table, dict) and item_id in table
               and name not in {'items', 'item_builds', 'trait_items'}}
    return {**summary, 'base_stats': deepcopy(item_stats.items[item_id]), 'effects': effects,
            'recipe': deepcopy(item_stats.item_builds.get(item_id)),
            'builds_into': [{'item_id': result, 'components': deepcopy(components)}
                            for result, components in sorted(item_stats.item_builds.items())
                            if item_id in components],
            'granted_trait': granted_trait, 'constraints': constraints, 'description': None,
            'unavailable_fields': ['description']}
