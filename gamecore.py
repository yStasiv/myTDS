from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from math import ceil
import random

class WeaponKind(Enum):
    MELEE = auto()
    RANGED = auto()

# Ймовірність, що ворог "зачепився" (контакт) у конкретний кадр (було ~0.2)
CONTACT_CHANCE = 0.08

# Раз на скільки секунд контакти завдають шкоди (раніше билося майже кожен кадр)
CONTACT_HIT_INTERVAL = 0.6  # секунди

# Імовірність удару під час “тика” контактного урону (було 0.5)
CONTACT_HIT_PROB = 0.35

# Сповільнюємо спавн ворогів (множник до інтервалів спавну веж)
SPAWN_RATE_MULT = 1.35  # 35% повільніше

# Трошки міцніша базова вагонетка (щоб менше тонула від РНГ)
BASE_CART_HP = 100  # було 100
BASE_CART_ARMOR = 0  # було 0

SEGMENT_REWARDS = [1, 2, 3, 4, 5, 6]  # 0–1..5–Boss
ENEMY_DMG_BY_SEGMENT = [(1,10),(5,15),(10,20),(15,25),(20,30),(25,40)]
BOSS_DMG = (100, 200)

BASE_COINS = 2000
CART_UPGRADE_COSTS = [50, 100, 150, 200]
TOWER_ENEMY_HP_MULT = [1.0, 1.2, 1.5, 1.9, 2.4]

def boss_hp(level_idx: int) -> int:
    hp = 10000
    for n in range(2, level_idx + 1):
        hp = round((hp + 10000) * 1.10)
    return hp

def weapon_upgrade_cost(i: int) -> int:
    return ceil(30 * (1.15 ** (i - 1)))

def weapon_tier_transition_cost() -> int:
    return sum(weapon_upgrade_cost(i) for i in range(1, 6))

@dataclass
class Weapon:
    name: str
    kind: WeaponKind
    base_damage: float
    shots_per_second: float
    tier: int = 1
    upgrades_done_in_tier: int = 0
    bonus_armor: int = 0
    regen_percent_per_10s: float = 0.0

    def current_dps(self) -> float:
        return self.base_damage * self.shots_per_second

    def upgrade(self, coins: int) -> int:
        if self.tier >= 5 and self.upgrades_done_in_tier >= 5:
            return coins
        # upgrade within tier
        if self.upgrades_done_in_tier < 5:
            cost = weapon_upgrade_cost(self.upgrades_done_in_tier + 1)
            if coins >= cost:
                coins -= cost
                if self.kind == WeaponKind.RANGED:
                    self.shots_per_second *= 1.10
                else:
                    self.base_damage *= 1.10
                self.upgrades_done_in_tier += 1
            return coins
        # transition
        cost = weapon_tier_transition_cost()
        if self.tier < 5 and coins >= cost:
            coins -= cost
            self.tier += 1
            self.upgrades_done_in_tier = 0
        return coins

    def pick_bonus(self, bonus: str):
        if bonus == "armor":
            self.bonus_armor += 1
        elif bonus == "regen":
            self.regen_percent_per_10s += 0.5

@dataclass
class Cart:
    max_hp: int = BASE_CART_HP
    armor: int = BASE_CART_ARMOR
    hp: int = BASE_CART_HP
    base_speed: float = 1.0
    contacts: int = 0
    level: int = 0

    def speed(self) -> float:
        return max(0.3 * self.base_speed, self.base_speed * (1 - 0.10 * self.contacts))

    def take_damage(self, dmg: int):
        reduced = max(0, dmg - self.armor)
        self.hp = max(0, self.hp - reduced)

    def upgrade(self, coins: int) -> int:
        if self.level >= 4:
            return coins
        cost = CART_UPGRADE_COSTS[self.level]
        if coins >= cost:
            coins -= cost
            self.level += 1
            self.max_hp = int(self.max_hp * 1.20)
            self.hp = self.max_hp
            self.armor += 1
        return coins

@dataclass
class Enemy:
    hp: float
    alive: bool = True
    in_contact: bool = False

@dataclass
class Tower:
    idx: int
    hp: float
    spawn_interval: float
    time_since_spawn: float = 0.0
    destroyed: bool = False

    def try_spawn(self, dt: float, mk_enemy_hp) -> list[Enemy]:
        if self.destroyed:
            return []
        self.time_since_spawn += dt
        out = []
        while self.time_since_spawn >= self.spawn_interval:
            self.time_since_spawn -= self.spawn_interval
            out.append(Enemy(hp=mk_enemy_hp(self.idx)))
        return out

@dataclass
class LevelState:
    level_idx: int
    cart: Cart
    base_weapon: Weapon
    chosen_weapon: Weapon
    coins: int = BASE_COINS
    crystals: int = 0
    segment_idx: int = 0
    towers: list[Tower] = field(default_factory=list)
    enemies: list[Enemy] = field(default_factory=list)
    boss_hp: float = 0
    boss_alive: bool = True
    paused_at_tower: bool = False
    status: str = ""
    contact_damage_timer: float = 0.0


    def init_level(self):
        self.boss_hp = boss_hp(self.level_idx)
        self.towers = [
            Tower(
                idx=i,
                hp=int(300 * TOWER_ENEMY_HP_MULT[i]),
                spawn_interval=max(0.8 - 0.1 * i, 0.3) * SPAWN_RATE_MULT
            )
            for i in range(5)
        ]
    def current_reward(self) -> int:
        return SEGMENT_REWARDS[min(self.segment_idx, 5)]

    def enemy_damage_range(self):
        return ENEMY_DMG_BY_SEGMENT[min(self.segment_idx, 5)]

def can_damage_tower(state, tower_idx: int) -> bool:
    return state.paused_at_tower and state.segment_idx == (tower_idx + 1)

def can_damage_boss(state) -> bool:
    return state.paused_at_tower and state.segment_idx == 6

def simulate_tick(state: LevelState, dt: float):
    # spawn from nearest alive tower
    active_tower = next((t for t in state.towers if not t.destroyed), None)
    if active_tower:
        def mk_enemy_hp(tower_idx: int) -> int:
            base = 30
            return int(base * TOWER_ENEMY_HP_MULT[tower_idx])
        state.enemies.extend(active_tower.try_spawn(dt, mk_enemy_hp))

    def damage_enemies(dmg: float, contacts_only=False):
        targets = [e for e in state.enemies if e.alive and (e.in_contact if contacts_only else True)]
        if not targets:
            return
        split = max(1, min(3, len(targets)))
        for e in targets[:split]:
            e.hp -= dmg / split
            if e.hp <= 0 and e.alive:
                e.alive = False
                state.coins += state.current_reward()

    # apply DPS
    dmg_base = state.base_weapon.current_dps() * dt
    dmg_chosen = state.chosen_weapon.current_dps() * dt

    # melee = contacts; ranged = free; 50% tower when paused
    active_tower = next((t for t in state.towers if not t.destroyed), None)
    # base weapon: завжди вільний у прикладі
    damage_enemies(dmg_base, contacts_only=False)

    if state.chosen_weapon.kind == WeaponKind.MELEE:
        damage_enemies(dmg_chosen, contacts_only=True)
    else:
        if state.paused_at_tower and active_tower and not active_tower.destroyed and random.random() < 0.5:
            active_tower.hp -= dmg_chosen
        else:
            damage_enemies(dmg_chosen, contacts_only=False)

    # cleanup enemies
    for e in state.enemies:
        if e.alive and e.hp <= 0:
            e.alive = False
            state.coins += state.current_reward()
    state.enemies = [e for e in state.enemies if e.alive]

    # tower destruction
    if active_tower and active_tower.hp <= 0 and not active_tower.destroyed:
        active_tower.destroyed = True
        state.enemies.clear()
        state.paused_at_tower = False
        state.segment_idx = min(state.segment_idx + 1, 5)
        state.status = f"Вежа {active_tower.idx+1} знищена"

    # contacts (псевдо)
    for e in state.enemies:
        e.in_contact = random.random() < CONTACT_CHANCE
    state.cart.contacts = sum(1 for e in state.enemies if e.in_contact)

    # contact damage
    state.contact_damage_timer += dt
    if state.contact_damage_timer >= CONTACT_HIT_INTERVAL:
        state.contact_damage_timer -= CONTACT_HIT_INTERVAL
        low, high = state.enemy_damage_range()
        for e in state.enemies:
            if e.in_contact and random.random() < CONTACT_HIT_PROB:
                state.cart.take_damage(random.randint(low, high))

    # pause at tower (плейсхолдер)
    state.paused_at_tower = bool(active_tower and not active_tower.destroyed and random.random() < 0.4)

    # boss phase
    if not any(not t.destroyed for t in state.towers) and state.boss_alive:
        state.segment_idx = 5
        total_dps = (state.base_weapon.current_dps() + state.chosen_weapon.current_dps())
        state.boss_hp -= total_dps * dt
        if random.random() < 0.3:
            state.cart.take_damage(random.randint(*BOSS_DMG))
        if state.boss_hp <= 0:
            state.boss_alive = False
            state.crystals += 1
            state.status = "Бос знищений!"
