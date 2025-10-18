
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import random

# ---- Balancing constants ----
BASE_COINS = 200
ENEMY_CONTACT_CM = 2.0
CART_BASE_HP = 120
CART_BASE_ARMOR = 0.0

# Towers HP multipliers (from 1st to 5th)
TOWER_ENEMY_HP_MULT = [1.0, 1.3, 1.6, 2.0, 2.6]
TOWER_BASE_HP = 320.0

# Waves
SPAWN_GAP_RANGE = (0.8, 3.0)
BURST_COUNT_RANGE = (2, 4)
BURST_INTERVAL = 0.22
MINIBOSS_CHANCE = 0.12

# World
TOWER_WORLD_X = [150.0, 300.0, 450.0, 700.0, 950.0]

# Boss HP scaling
def boss_hp_for_level(level_idx: int) -> float:
    hp = 10000.0
    if level_idx <= 1:
        return hp
    for i in range(2, level_idx + 1):
        hp = hp + 10000.0
        hp *= 1.10
    return hp

class WeaponKind(Enum):
    RANGED = "ranged"
    MELEE = "melee"

# --- Enemy type registry ---
ENEMY_TYPES: Dict[str, Dict] = {
    "slime":   {"sprite":"slime.png",  "hp":18.0,  "speed_cm_s":0.35, "attack_dps":5.0,  "kb_resist":0.2},
    "goblin":  {"sprite":"goblin.png", "hp":28.0,  "speed_cm_s":0.60, "attack_dps":8.0,  "kb_resist":0.3},
    "bat":     {"sprite":"bat.png",    "hp":14.0,  "speed_cm_s":0.90, "attack_dps":6.0,  "kb_resist":0.1},
    "brute":   {"sprite":"brute.png",  "hp":60.0,  "speed_cm_s":0.30, "attack_dps":15.0, "kb_resist":0.6},
    "miniboss":{"sprite":"miniboss.png","hp":220.0,"speed_cm_s":0.28, "attack_dps":22.0, "kb_resist":0.7},
}

# --- Per-tower enemy pools ---
TOWER_ENEMY_POOLS: Dict[int, List[str]] = {
    0: ["slime", "goblin"],
    1: ["slime", "goblin", "bat"],
    2: ["goblin", "bat", "brute"],
    3: ["bat", "brute"],
    4: ["brute", "goblin", "bat"],
}

# --- Weapon registry with attack/reload cycles ---
WEAPON_TYPES: Dict[str, Dict] = {
    # BASE SET
    "Пила": {  # continuous melee
        "kind": WeaponKind.MELEE,
        "base_damage_per_tick": 2.0,
        "range_cm": 5.0,
        "attack_duration": 9999.0,
        "reload_duration": 0.0,
        "tick_hz": 12.0,
        "upgrade_base_cost": 25,
        "upgrade_growth": 1.12,
    },
    "Кулемет": {
        "kind": WeaponKind.RANGED,
        "base_damage_per_tick": 0.9,
        "range_cm": 42.0,
        "attack_duration": 0.30,
        "reload_duration": 0.10,
        "tick_hz": 16.0,
        "upgrade_base_cost": 35,
        "upgrade_growth": 1.15,
    },
    "Вогнемет": {
        "kind": WeaponKind.RANGED,
        "base_damage_per_tick": 0.5,
        "range_cm": 26.0,
        "attack_duration": 0.50,
        "reload_duration": 0.20,
        "tick_hz": 22.0,
        "upgrade_base_cost": 30,
        "upgrade_growth": 1.14,
    },
    "Лазер": {
        "kind": WeaponKind.RANGED,
        "base_damage_per_tick": 0.7,
        "range_cm": 48.0,
        "attack_duration": 0.40,
        "reload_duration": 0.20,
        "tick_hz": 28.0,
        "upgrade_base_cost": 40,
        "upgrade_growth": 1.18,
        "pierce": 3,
    },
    # NEW ADDITIONS
    "Молот": {  # melee heavy slow hits (short bursts, strong ticks)
        "kind": WeaponKind.MELEE,
        "base_damage_per_tick": 6.0,
        "range_cm": 6.0,
        "attack_duration": 0.30,
        "reload_duration": 0.40,
        "tick_hz": 6.0,
        "upgrade_base_cost": 32,
        "upgrade_growth": 1.13,
        "kb_bonus": 20.0,
    },
    "Дробовик": {  # short range cone, multiple pellets
        "kind": WeaponKind.RANGED,
        "base_damage_per_tick": 0.6,   # per pellet
        "range_cm": 24.0,
        "attack_duration": 0.25,
        "reload_duration": 0.35,
        "tick_hz": 10.0,
        "upgrade_base_cost": 34,
        "upgrade_growth": 1.16,
        "pellets": 4,
    },
    "Плазма": {  # long range, slow but heavy ticks
        "kind": WeaponKind.RANGED,
        "base_damage_per_tick": 3.0,
        "range_cm": 50.0,
        "attack_duration": 0.15,
        "reload_duration": 0.45,
        "tick_hz": 6.0,
        "upgrade_base_cost": 42,
        "upgrade_growth": 1.17,
        "splash": 0.4,  # dmg to second target
    },
}

# helper lists
MELEE_WEAPONS = [name for name, spec in WEAPON_TYPES.items() if spec["kind"] == WeaponKind.MELEE]
RANGED_WEAPONS = [name for name, spec in WEAPON_TYPES.items() if spec["kind"] == WeaponKind.RANGED]

@dataclass
class Weapon:
    name: str
    kind: WeaponKind
    damage_per_tick: float
    range_cm: float
    attack_duration: float
    reload_duration: float
    tick_hz: float
    upgrade_base_cost: int
    upgrade_growth: float
    tier: int = 1
    upgrades_done_in_tier: int = 0

    _phase_time: float = 0.0
    _attacking: bool = True
    _tick_cd: float = 0.0

    @classmethod
    def from_type(cls, name: str):
        spec = WEAPON_TYPES[name]
        return cls(
            name=name,
            kind=spec["kind"],
            damage_per_tick=spec["base_damage_per_tick"],
            range_cm=spec["range_cm"],
            attack_duration=spec["attack_duration"],
            reload_duration=spec["reload_duration"],
            tick_hz=spec["tick_hz"],
            upgrade_base_cost=spec["upgrade_base_cost"],
            upgrade_growth=spec["upgrade_growth"],
        )

    def _next_upgrade_cost(self) -> int:
        k = self.upgrades_done_in_tier
        if k < 5:
            return int(round(self.upgrade_base_cost * (self.upgrade_growth ** k)))
        total = sum(int(round(self.upgrade_base_cost * (self.upgrade_growth ** i))) for i in range(5))
        return total

    def upgrade(self, coins: int) -> int:
        cost = self._next_upgrade_cost()
        if coins < cost:
            return coins
        coins -= cost
        if self.upgrades_done_in_tier < 5:
            self.upgrades_done_in_tier += 1
            self.damage_per_tick *= 1.12
            self.tick_hz *= 1.04
            if self.kind == WeaponKind.RANGED:
                self.range_cm *= 1.02
        else:
            self.tier += 1
            self.upgrades_done_in_tier = 0
        return coins

    def advance_cycle(self, dt: float):
        if self.attack_duration >= 9999.0:
            self._attacking = True
            return
        self._phase_time -= dt
        if self._phase_time <= 0:
            if self._attacking:
                self._attacking = False
                self._phase_time = self.reload_duration
            else:
                self._attacking = True
                self._phase_time = self.attack_duration

    def reset_cycle(self):
        self._attacking = True
        self._phase_time = self.attack_duration if self.attack_duration < 9999.0 else 0.0
        self._tick_cd = 0.0

@dataclass
class Cart:
    level: int = 0
    max_hp: float = CART_BASE_HP
    hp: float = CART_BASE_HP
    armor: float = CART_BASE_ARMOR

    def upgrade(self, coins: int) -> int:
        costs = [50, 100, 150, 200]
        if self.level >= 4 or coins < costs[self.level]:
            return coins
        coins -= costs[self.level]
        self.level += 1
        self.max_hp = int(self.max_hp * 1.35 + 15)
        self.armor += 0.5
        self.hp = self.max_hp
        return coins

@dataclass
class Enemy:
    source_tower_idx: int
    type_key: str
    wx: float
    hp: float
    max_hp: float
    speed_cm_s: float
    attack_dps: float
    kb_resist: float
    spawn_t: float = 0.25
    dying: bool = False
    death_t: float = 0.35
    dash_ready: bool = True
    dash_t: float = 0.0
    knockback_vx: float = 0.0
    hit_flash_t: float = 0.0
    fx_dash_trigger: bool = False
    fx_death_trigger: bool = False
    fx_hit_trigger: bool = False

    @classmethod
    def from_type(cls, tower_idx: int, type_key: str, wx: float, hp_mult: float = 1.0):
        spec = ENEMY_TYPES[type_key]
        base_hp = spec["hp"] * hp_mult
        return cls(
            source_tower_idx=tower_idx,
            type_key=type_key,
            wx=wx,
            hp=base_hp,
            max_hp=base_hp,
            speed_cm_s=spec["speed_cm_s"],
            attack_dps=spec["attack_dps"],
            kb_resist=spec["kb_resist"],
        )

@dataclass
class Tower:
    idx: int
    hp: float
    destroyed: bool = False
    is_tower: bool = True

@dataclass
class LevelState:
    level_idx: int
    cart: Cart
    base_weapon: Weapon
    chosen_weapon: Weapon
    coins: int = BASE_COINS
    crystals: int = 0

    towers: List[Tower] = field(default_factory=list)
    boss_hp: float = 0.0
    boss_alive: bool = True
    enemies: List[Enemy] = field(default_factory=list)

    segment_idx: int = 0
    paused_at_tower: bool = False

    # spawning (waves)
    spawn_gap_t: float = 0.0
    burst_count_left: int = 0
    burst_timer: float = 0.0
    burst_type_key: Optional[str] = None

    # weapon pool for current LEVEL (persist across attempts)
    level_weapon_pool: List[str] = field(default_factory=list)

    def init_level(self):
        self.towers = [Tower(i, TOWER_BASE_HP * TOWER_ENEMY_HP_MULT[i]) for i in range(5)]
        self.boss_hp = boss_hp_for_level(self.level_idx)
        self.boss_alive = True
        self.enemies.clear()
        self.segment_idx = 0
        self.paused_at_tower = False
        self.spawn_gap_t = random.uniform(*SPAWN_GAP_RANGE)
        self.burst_count_left = 0
        self.burst_timer = 0.0
        self.burst_type_key = None
        self.base_weapon.reset_cycle()
        self.chosen_weapon.reset_cycle()

def can_damage_tower(state, tower_idx: int) -> bool:
    return state.paused_at_tower and state.segment_idx == (tower_idx + 1)

def can_damage_boss(state) -> bool:
    return state.paused_at_tower and state.segment_idx == 6

# --- Core simulation tick ---
def simulate_tick(state: LevelState, dt: float):
    if dt <= 0:
        return

    # 1) Waves
    active_idx = next((t.idx for t in state.towers if not t.destroyed), None)
    if active_idx is not None and state.cart.hp > 0 and state.boss_alive:
        if state.burst_count_left > 0:
            state.burst_timer -= dt
            if state.burst_timer <= 0:
                spawn_x = TOWER_WORLD_X[active_idx]
                hp_mult = TOWER_ENEMY_HP_MULT[active_idx]
                type_key = state.burst_type_key or random.choice(TOWER_ENEMY_POOLS.get(active_idx, list(ENEMY_TYPES.keys())))
                state.enemies.append(Enemy.from_type(active_idx, type_key, wx=spawn_x, hp_mult=hp_mult))
                state.burst_count_left -= 1
                state.burst_timer = BURST_INTERVAL if state.burst_count_left > 0 else 0.0
        else:
            state.spawn_gap_t -= dt
            if state.spawn_gap_t <= 0:
                if random.random() < MINIBOSS_CHANCE and "miniboss" in ENEMY_TYPES:
                    state.burst_type_key = "miniboss"; state.burst_count_left = 1
                else:
                    pool = TOWER_ENEMY_POOLS.get(active_idx, list(ENEMY_TYPES.keys()))
                    state.burst_type_key = random.choice(pool)
                    lo, hi = BURST_COUNT_RANGE
                    state.burst_count_left = random.randint(lo, hi)
                state.burst_timer = 0.0
                state.spawn_gap_t = random.uniform(*SPAWN_GAP_RANGE)

    # 2) Enemies update
    cart_wx = getattr(state, "_runtime_cart_wx", 0.0)
    for e in state.enemies[:]:
        if e.dying:
            e.death_t -= dt
            if e.death_t <= 0:
                try: state.enemies.remove(e)
                except ValueError: pass
            continue

        if e.spawn_t > 0:
            e.spawn_t = max(0.0, e.spawn_t - dt)

        if e.type_key == "bat" and e.dash_ready and (e.wx - cart_wx) <= 30.0:
            e.dash_ready = False; e.dash_t = 0.6; e.fx_dash_trigger = True

        base_v = -e.speed_cm_s
        if e.dash_t > 0:
            e.dash_t -= dt; base_v *= 2.2

        base_v += e.knockback_vx
        if e.knockback_vx != 0.0:
            decay = 6.0 * dt
            if e.knockback_vx > 0: e.knockback_vx = max(0.0, e.knockback_vx - decay)
            else: e.knockback_vx = min(0.0, e.knockback_vx + decay)

        e.wx += base_v * dt
        if e.wx < cart_wx + 0.5: e.wx = cart_wx + 0.5

    # 3) Contact DPS
    total_dps = 0.0
    for e in state.enemies:
        if e.dying: continue
        if (e.wx - cart_wx) <= ENEMY_CONTACT_CM + 1e-6:
            total_dps += e.attack_dps
    if total_dps > 0:
        state.cart.hp -= total_dps * dt

    # 4) Weapons cycle advance
    state.base_weapon.advance_cycle(dt)
    state.chosen_weapon.advance_cycle(dt)

    # 5) Weapons damage
    def weapon_tick(wpn: Weapon):
        wpn._tick_cd -= dt
        if not wpn._attacking or wpn._tick_cd > 0:
            return
        wpn._tick_cd += max(0.0001, 1.0 / max(0.5, wpn.tick_hz))

        rng = wpn.range_cm
        in_range = [e for e in state.enemies if not e.dying and (e.wx - cart_wx) <= rng + 1e-6 and e.wx >= cart_wx]
        if not in_range:
            return

        def apply_enemy_damage(target_enemy, dmg, kb_cm):
            target_enemy.hp -= dmg
            target_enemy.hit_flash_t = 0.08
            target_enemy.fx_hit_trigger = True
            if target_enemy.hp <= 0 and not target_enemy.dying:
                target_enemy.dying = True
                target_enemy.death_t = 0.35
                target_enemy.fx_death_trigger = True
                award = max(1, state.segment_idx)
                state.coins += award
            else:
                resist = target_enemy.kb_resist
                push = max(0.0, kb_cm) * (1.0 - resist)
                target_enemy.knockback_vx -= push

        dmg = wpn.damage_per_tick

        if wpn.kind == WeaponKind.MELEE:
            # melee: if 'Молот' add extra knockback
            target = min(in_range, key=lambda e: e.wx)
            kb = 14.0 + float(WEAPON_TYPES[wpn.name].get("kb_bonus", 0.0))
            apply_enemy_damage(target, dmg, kb_cm=kb)
        else:
            if wpn.name == "Лазер":
                pierce = int(WEAPON_TYPES[wpn.name].get("pierce", 1))
                for t in sorted(in_range, key=lambda e: e.wx)[:pierce]:
                    apply_enemy_damage(t, dmg, kb_cm=6.0)
            elif wpn.name == "Вогнемет":
                targets = sorted(in_range, key=lambda e: e.wx)[:2]
                for i, t in enumerate(targets):
                    apply_enemy_damage(t, dmg * (1.0 if i == 0 else 0.6), kb_cm=5.0)
            elif wpn.name == "Дробовик":
                pellets = int(WEAPON_TYPES[wpn.name].get("pellets", 3))
                targets = sorted(in_range, key=lambda e: e.wx)[:pellets]
                for i, t in enumerate(targets):
                    apply_enemy_damage(t, dmg * (1.0 - 0.1*i), kb_cm=7.0)
            elif wpn.name == "Плазма":
                targets = sorted(in_range, key=lambda e: e.wx)[:2]
                if targets:
                    apply_enemy_damage(targets[0], dmg, kb_cm=9.0)
                    if len(targets) > 1:
                        apply_enemy_damage(targets[1], dmg * float(WEAPON_TYPES[wpn.name].get("splash", 0.3)), kb_cm=6.0)
            else:
                target = min(in_range, key=lambda e: e.wx)
                apply_enemy_damage(target, dmg, kb_cm=8.0)

    weapon_tick(state.base_weapon)
    weapon_tick(state.chosen_weapon)
