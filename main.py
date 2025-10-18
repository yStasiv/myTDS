
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.properties import StringProperty, ObjectProperty, NumericProperty, ListProperty, DictProperty
from kivy.core.image import Image as CoreImage
from kivy.core.audio import SoundLoader

import json, os, random

from gamecore import (
    Weapon, WeaponKind, Cart, LevelState, simulate_tick,
    TOWER_ENEMY_HP_MULT, BASE_COINS, ENEMY_TYPES,
    WEAPON_TYPES, MELEE_WEAPONS, RANGED_WEAPONS
)

SAVE_PATH = "save.json"

def load_save():
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"exp": 0, "crystals": 0}

def save_save(data):
    try:
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

Builder.load_file('ui.kv')

WORLD_LENGTH = 1000.0
TOWER_XS = [150.0, 300.0, 450.0, 700.0, 950.0]
BOSS_X = 1000.0
STOP_BEFORE = 35.0
CAMERA_RANGE = (-50.0, 450.0)
LOD_DIST = 180.0

STOP_POSITIONS = [0.0] + [max(0.0, x - STOP_BEFORE) for x in TOWER_XS + [BOSS_X]]

def roll_level_weapon_pool() -> list[str]:
    melee = random.choice(MELEE_WEAPONS)
    ranged_choices = random.sample(RANGED_WEAPONS, k=2)
    return [melee] + ranged_choices

class TopProgress(Widget):
    ratio = NumericProperty(0.0)
    segments = NumericProperty(6)
    def on_size(self, *a): self._redraw()
    def on_pos(self, *a): self._redraw()
    def on_ratio(self, *a): self._redraw()
    def _redraw(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1,1,1,0.12); Rectangle(pos=self.pos, size=self.size)
            seg_w = self.width / float(self.segments)
            for i in range(1, self.segments):
                x = self.x + seg_w * i
                Color(1,1,1,0.24); Line(points=[x, self.y, x, self.top], width=1)
            Color(0.3,0.8,1.0,0.95)
            w = max(0.0, min(1.0, self.ratio)) * self.width
            Rectangle(pos=self.pos, size=(w, self.height))

class Playfield(Widget):
    left_pad = NumericProperty(48)
    right_pad = NumericProperty(48)
    top_pad = NumericProperty(32)
    bottom_pad = NumericProperty(32)

    baseline_y = NumericProperty(0)
    world_len = NumericProperty(WORLD_LENGTH)
    towers_world_x = ListProperty(TOWER_XS[:])
    boss_world_x = NumericProperty(BOSS_X)

    tower_w = NumericProperty(24)
    tower_h = NumericProperty(96)

    destroy_flash = DictProperty({})
    destroyed_prev = DictProperty({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enemy_textures = {}
        for key, spec in ENEMY_TYPES.items():
            path = f"assets/{spec['sprite']}"
            try:
                self.enemy_textures[key] = CoreImage(path).texture
            except Exception:
                pass
        self.snd_dash = SoundLoader.load('assets/sfx/dash.wav') or None
        self.snd_death = SoundLoader.load('assets/sfx/death.wav') or None
        self.snd_hit = SoundLoader.load('assets/sfx/hit.wav') or None
        self._hit_cooldown = 0.0

    def on_size(self, *args):
        self._recalc_layout(); self._draw_background()

    def on_pos(self, *args):
        self._recalc_layout(); self._draw_background()

    def _recalc_layout(self):
        W, H = self.width, self.height
        self.baseline_y = self.y + self.bottom_pad + (H - self.top_pad - self.bottom_pad) * 0.35
        self.towers_world_x = TOWER_XS[:]
        self.boss_world_x   = BOSS_X
        usable_h = H - self.top_pad - self.bottom_pad
        self.tower_h = max(80, usable_h * 0.25)
        usable_w = W - self.left_pad - self.right_pad
        self.tower_w = max(18, min(36, usable_w * 0.02))

    def _draw_background(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.10, 0.12, 0.16, 1); Rectangle(pos=self.pos, size=self.size)

    def _camera_window(self, cart_wx: float):
        cam_min, cam_max = CAMERA_RANGE
        cam_w = cam_max - cam_min
        left = cart_wx + cam_min; right = cart_wx + cam_max
        if left < 0: right += -left; left = 0.0
        if right > self.world_len:
            shift = right - self.world_len
            left -= shift; right = self.world_len
            if left < 0: left = 0.0
        if right - left < cam_w:
            right = min(self.world_len, left + cam_w)
            left  = max(0.0, right - cam_w)
        return left, right, cam_w

    def _w2s(self, world_x: float, cam_left: float, cam_right: float) -> float:
        x0 = self.x + self.left_pad; x1 = self.right - self.right_pad
        if cam_right == cam_left: return x0
        t = (world_x - cam_left) / (cam_right - cam_left)
        return x0 + (x1 - x0) * t

    def cart_world_x(self, state, seg_progress: float, phase: str) -> float:
        nodes = STOP_POSITIONS
        seg = max(0, min(state.segment_idx, len(nodes)-2))
        x0, x1 = nodes[seg], nodes[seg+1]
        if phase != 'running' or state.paused_at_tower:
            return x0
        return x0 + (x1 - x0) * max(0.0, min(1.0, seg_progress))

    def _play_once(self, snd):
        if snd:
            try: snd.play()
            except Exception: pass

    def render(self, state, seg_progress: float, dt: float, phase: str):
        from gamecore import TOWER_ENEMY_HP_MULT, ENEMY_TYPES
        self.canvas.after.clear()
        with self.canvas.after:
            cart_wx = self.cart_world_x(state, seg_progress, phase)
            state._runtime_cart_wx = cart_wx

            cam_left, cam_right, _ = self._camera_window(cart_wx)

            Color(1,1,1,0.12)
            Line(points=[ self._w2s(cam_left, cam_left, cam_right), self.baseline_y,
                          self._w2s(cam_right, cam_left, cam_right), self.baseline_y ], width=1)

            for i, wx in enumerate(self.towers_world_x):
                if not (cam_left <= wx <= cam_right): continue
                tower = next((t for t in state.towers if t.idx == i), None)
                if not tower: continue

                was_destroyed = self.destroyed_prev.get(i, False)
                if tower.destroyed and not was_destroyed:
                    self.destroy_flash[i] = 0.25
                self.destroyed_prev[i] = tower.destroyed

                sx = self._w2s(wx, cam_left, cam_right)
                if tower.destroyed: Color(0.25,0.28,0.32,1)
                elif i == next((t.idx for t in state.towers if not t.destroyed), 4): Color(0.85,0.45,0.25,1)
                else: Color(0.55,0.62,0.72,1)
                Rectangle(pos=(sx - self.tower_w/2, self.baseline_y), size=(self.tower_w, self.tower_h))

                maxhp = 320 * TOWER_ENEMY_HP_MULT[i]
                hp = max(0.0, min(maxhp, tower.hp if not tower.destroyed else 0.0))
                ratio = (hp / maxhp) if maxhp > 0 else 0.0
                bar_w, bar_h = self.tower_w, 6
                Color(0.2,0.2,0.2,1); Rectangle(pos=(sx - bar_w/2, self.baseline_y - 10), size=(bar_w, bar_h))
                Color(0.2,0.8,0.25,1); Rectangle(pos=(sx - bar_w/2, self.baseline_y - 10), size=(bar_w*ratio, bar_h))

                if self.destroy_flash.get(i, 0) > 0:
                    a = max(0.0, min(1.0, self.destroy_flash[i] / 0.25))
                    Color(1,0.9,0.3,a)
                    Rectangle(pos=(sx - self.tower_w/2 - 4, self.baseline_y - 4), size=(self.tower_w + 8, self.tower_h + 8))
                    self.destroy_flash[i] = max(0.0, self.destroy_flash[i] - dt)

            if cam_left <= self.boss_world_x <= cam_right:
                sx = self._w2s(self.boss_world_x, cam_left, cam_right)
                Color(0.9,0.2,0.2,1)
                Rectangle(pos=(sx - self.tower_w/2, self.baseline_y + self.tower_h*0.15), size=(self.tower_w, self.tower_h*0.7))

            self._hit_cooldown = max(0.0, self._hit_cooldown - dt)
            for e in state.enemies[:]:
                ex = e.wx
                if not (cam_left <= ex <= cam_right): continue
                sx = self._w2s(ex, cam_left, cam_right)

                # sizes (fallback constants; can be expanded to size registry as before)
                base_size = 28
                draw_w = base_size; draw_h = base_size
                img_y = self.baseline_y - draw_h * 0.5
                tex = self.enemy_textures.get(e.type_key)

                if getattr(e, "fx_dash_trigger", False):
                    self._play_once(self.snd_dash); e.fx_dash_trigger = False
                if getattr(e, "fx_death_trigger", False):
                    self._play_once(self.snd_death); e.fx_death_trigger = False
                if getattr(e, "fx_hit_trigger", False) and self._hit_cooldown <= 0.0:
                    self._play_once(self.snd_hit); e.fx_hit_trigger = False; self._hit_cooldown = 0.05

                Color(1,1,1,1)
                if tex is not None:
                    Rectangle(texture=tex, pos=(sx - draw_w/2, img_y), size=(draw_w, draw_h))
                else:
                    Color(0.85,0.85,1.0,1); Ellipse(pos=(sx - 4, self.baseline_y - 4), size=(8, 8))

                if e.hit_flash_t > 0:
                    e.hit_flash_t = max(0.0, e.hit_flash_t - dt)
                    a = min(0.5, 0.5 * (e.hit_flash_t / 0.10 + 0.2))
                    Color(1.0, 1.0, 1.0, a)
                    Rectangle(pos=(sx - draw_w/2, img_y), size=(draw_w, draw_h))

                ratio = max(0.0, min(1.0, e.hp / e.max_hp)) if e.max_hp > 0 else 0.0
                bar_w, bar_h = 24, 4
                bar_y = img_y + draw_h + 4
                Color(0.1,0.1,0.1,0.9); Rectangle(pos=(sx - bar_w/2, bar_y), size=(bar_w, bar_h))
                Color(0.2,0.8,0.25,0.95); Rectangle(pos=(sx - bar_w/2, bar_y), size=(bar_w * ratio, bar_h))

            if cam_left <= cart_wx <= cam_right:
                cx = self._w2s(cart_wx, cam_left, cam_right); cy = self.baseline_y - 18
                Color(0.95,0.95,0.95,1); Rectangle(pos=(cx - 18, cy - 12), size=(36, 24))
                Color(0.3,0.3,0.3,1); Ellipse(pos=(cx - 16, cy - 18), size=(10, 10)); Ellipse(pos=(cx + 6, cy - 18), size=(10, 10))

            if phase == 'build':
                Color(1,1,1,0.05); Rectangle(pos=self.pos, size=self.size)

class GameRoot(BoxLayout):
    external_state = ObjectProperty(allownone=True, rebind=True)

    status_text = StringProperty("")
    phase = StringProperty("build")
    state = ObjectProperty(allownone=True, rebind=True)

    cart_cost_text = StringProperty("Апгр. Вагонетки")
    weapon_cost_text = StringProperty("Апгр. Зброї")
    weapon_select_text = StringProperty("Зброя: -")
    progress_pct_text = StringProperty("0.0%")

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.seg_progress = 0.0
        self.time_scale = 1.0
        self.cart_speed_cm_s = 1.0
        self._display_ratio = 0.0
        self._ev = Clock.schedule_interval(self.update, 1/24)
        self._level_weapon_pool = []  # 1 melee + 2 ranged (per LEVEL)
        self._weapon_idx = 0
        self._ensure_state_initial()

    def on_external_state(self, *_):
        if self.external_state is not None:
            self.state = self.external_state
            self._enter_build_phase("Режим добудови: оберіть/покращіть зброю та натисніть 'Старт'")

    def _roll_pool_for_level(self):
        # roll only if pool is empty (i.e., new level, not attempt)
        if not self._level_weapon_pool:
            self._level_weapon_pool = roll_level_weapon_pool()
        return self._level_weapon_pool

    def _ensure_state_initial(self):
        if self.state is None:
            # Roll weapon pool for this LEVEL and set default chosen from it
            pool = roll_level_weapon_pool()
            self._level_weapon_pool = pool[:]
            base_weapon = Weapon.from_type("Пила")  # базова «вагонетка»
            chosen_weapon = Weapon.from_type(pool[0])  # перша з пулу
            self.state = LevelState(level_idx=1, cart=Cart(), base_weapon=base_weapon, chosen_weapon=chosen_weapon, coins=BASE_COINS)
            self.state.level_weapon_pool = pool[:]
            self.state.init_level()
        else:
            # if loading from snapshot but pool not set, mirror from state
            self._level_weapon_pool = self.state.level_weapon_pool[:] if self.state.level_weapon_pool else roll_level_weapon_pool()
            if not self.state.chosen_weapon:
                self.state.chosen_weapon = Weapon.from_type(self._level_weapon_pool[0])
        self.weapon_select_text = f"Зброя: {self.state.chosen_weapon.name}"
        self._enter_build_phase("Режим добудови: оберіть/покращіть зброю та натисніть 'Старт'")

    def _enter_build_phase(self, msg):
        self.phase = 'build'
        self.seg_progress = 0.0
        self.status_text = msg
        self._refresh_prices_text()

    def _cart_next_cost(self):
        lvl = getattr(self.state.cart, "level", 0)
        costs = [50,100,150,200]
        if lvl >= 4: return None
        return costs[lvl]

    def _weapon_next_cost(self):
        return self.state.chosen_weapon._next_upgrade_cost()

    def _refresh_prices_text(self):
        c = self._cart_next_cost()
        self.cart_cost_text = f"Вагонетка +1 ({c} монет)" if c is not None else "Вагонетка MAX"
        w = self._weapon_next_cost()
        self.weapon_cost_text = f"Зброя + ({w} монет)" if w is not None else "Зброя MAX"
        self.weapon_select_text = f"Зброя: {self.state.chosen_weapon.name}"

    def _reinit_for_attempt(self):
        # preserve coins, cart, and weapon instances with their upgrades
        coins = self.state.coins
        cart = self.state.cart
        base_w = self.state.base_weapon
        chosen_w = self.state.chosen_weapon
        lvl = self.state.level_idx
        pool = self.state.level_weapon_pool[:] if self.state.level_weapon_pool else self._level_weapon_pool[:]
        self.state = LevelState(level_idx=lvl, cart=cart, base_weapon=base_w, chosen_weapon=chosen_w, coins=coins, crystals=self.state.crystals)
        self.state.level_weapon_pool = pool[:]
        self.state.init_level()
        self.state.segment_idx = 0
        self.state.paused_at_tower = False
        self.seg_progress = 0.0
        self._display_ratio = 0.0
        self._level_weapon_pool = pool[:]

    def start_run(self):
        if self.phase != 'build': return
        self._reinit_for_attempt()
        self.state.cart.hp = self.state.cart.max_hp
        self.phase = 'running'
        self.status_text = "Поїхали!"

    def surrender(self):
        self._enter_build_phase("Здача: без винагород. Монети доступні для добудови.")
        self.state.enemies.clear()
        self.state.cart.hp = self.state.cart.max_hp
        self.state.paused_at_tower = False

    def back_to_menu(self):
        App.get_running_app().save_level_snapshot(self.state)
        App.get_running_app().goto_menu()

    def upgrade_cart(self):
        if self.phase != 'build':
            self.status_text = "Покращення доступні лише у режимі добудови"; return
        before = self.state.coins
        self.state.coins = self.state.cart.upgrade(self.state.coins)
        spent = before - self.state.coins
        if spent > 0:
            self.status_text = f"Вагонетка +1 рівень (-{spent} монет)"
        self._refresh_prices_text()

    def upgrade_weapon(self):
        if self.phase != 'build':
            self.status_text = "Покращення доступні лише у режимі добудови"; return
        before = self.state.coins
        self.state.coins = self.state.chosen_weapon.upgrade(self.state.coins)
        spent = before - self.state.coins
        if spent > 0:
            self.status_text = f"Зброя покращена (-{spent} монет)"
        self._refresh_prices_text()

    def cycle_speed(self):
        self.time_scale = {1.0: 2.0, 2.0: 4.0}.get(self.time_scale, 1.0)
        self.status_text = f"Швидкість гри x{int(self.time_scale)}"

    def cycle_weapon(self):
        if self.phase != 'build':
            self.status_text = "Зміну зброї дозволено лише у режимі добудови"; return
        pool = self.state.level_weapon_pool[:] if self.state.level_weapon_pool else self._roll_pool_for_level()
        self._level_weapon_pool = pool[:]
        # IMPORTANT: when cycling, we **do not** reset upgrades of current weapon unless you pick another
        idx = pool.index(self.state.chosen_weapon.name) if self.state.chosen_weapon.name in pool else 0
        idx = (idx + 1) % len(pool)
        name = pool[idx]
        # create a new instance for the newly selected weapon (its upgrades start fresh)
        self.state.chosen_weapon = Weapon.from_type(name)
        self.status_text = f"Обрано: {name} (пул рівня)"
        self._refresh_prices_text()

    def _advance_segment_if_needed(self):
        if self.seg_progress >= 1.0:
            self.seg_progress = 0.0
            self.state.segment_idx = min(self.state.segment_idx + 1, 6)
            if self.state.segment_idx <= 6:
                self.state.paused_at_tower = True

    def _maybe_unpause_if_cleared(self):
        if not self.state.paused_at_tower:
            return
        node = self.state.segment_idx
        if node == 0:
            return
        if 1 <= node <= 5:
            tower_idx = node - 1
            if self.state.towers[tower_idx].destroyed:
                self.state.paused_at_tower = False
        elif node == 6:
            if not self.state.boss_alive:
                self.state.paused_at_tower = False

    def update(self, dt):
        self.ids.lbl_coins.text = f"Монети: {self.state.coins}"
        self.ids.lbl_hp.text = f"HP: {self.state.cart.hp}/{self.state.cart.max_hp}"
        self.ids.lbl_level.text = f"Рівень: {self.state.level_idx}"
        self.ids.lbl_segment.text = f"Сегмент: {self.state.segment_idx}"

        self._refresh_prices_text()

        cart_wx = self.ids.playfield.cart_world_x(self.state, self.seg_progress, self.phase)
        ratio = max(0.0, min(1.0, cart_wx / self.ids.playfield.world_len))
        alpha = 0.25
        if not hasattr(self, "_display_ratio"): self._display_ratio = 0.0
        self._display_ratio = self._display_ratio + (ratio - self._display_ratio) * alpha
        self.ids.top_progress.ratio = self._display_ratio
        self.ids.lbl_progress.text = f"{self._display_ratio*100:.1f}%"

        if self.phase == 'running':
            scaled_dt = dt * self.time_scale
            simulate_tick(self.state, scaled_dt)

            nodes = [0.0] + [max(0.0, x - 35.0) for x in TOWER_XS + [BOSS_X]]
            seg = max(0, min(self.state.segment_idx, len(nodes)-2))
            seg_len = max(1e-6, (nodes[seg+1] - nodes[seg]))
            if not self.state.paused_at_tower and self.state.segment_idx < 6:
                self.seg_progress += (self.cart_speed_cm_s / seg_len) * scaled_dt
                self._advance_segment_if_needed()
            else:
                self._maybe_unpause_if_cleared()

            if self.state.cart.hp <= 0:
                self._enter_build_phase("Поразка. Монети збережені — добудуй і спробуй знову.")
                self.state.enemies.clear()
                self.state.cart.hp = self.state.cart.max_hp
                self.state.paused_at_tower = False

            if not self.state.boss_alive and self.state.segment_idx >= 6:
                self._on_victory(); return

        vis_dt = dt * (self.time_scale if self.phase == 'running' else 1.0)
        self.ids.playfield.render(self.state, self.seg_progress, vis_dt, self.phase)

    def _on_victory(self):
        data = load_save()
        data["exp"] = int(data.get("exp", 0)) + 10
        data["crystals"] = int(data.get("crystals", 0)) + 1
        save_save(data)
        self.status_text = "Перемога! Кристал +1, EXP +10"
        app = App.get_running_app()
        app.clear_level_snapshot()
        app.goto_menu()

class MenuScreen(Screen):
    exp_text = StringProperty("EXP: 0")
    def on_pre_enter(self, *args):
        data = load_save()
        self.exp_text = f"EXP: {int(data.get('exp', 0))} | \u2727 {int(data.get('crystals', 0))}"

class GameScreen(Screen):
    incoming_state = ObjectProperty(allownone=True, rebind=True)

class ShopScreen(Screen): pass
class SettingsScreen(Screen): pass
class RootManager(ScreenManager): pass

class TowerCartTDApp(App):
    def build(self):
        self.title = "TowerCartTD"
        self.sm = RootManager(transition=FadeTransition(duration=0.15))
        self._level_snapshot = None
        return self.sm

    def save_level_snapshot(self, state): self._level_snapshot = state
    def clear_level_snapshot(self): self._level_snapshot = None

    def goto_menu(self):
        if self.sm.has_screen('menu'):
            self.sm.get_screen('menu').on_pre_enter()
        self.sm.current = 'menu'

    def start_game_continue(self):
        if self.sm.has_screen('game'):
            self.sm.remove_widget(self.sm.get_screen('game'))
        gs = GameScreen(name='game')
        if self._level_snapshot is None:
            # new LEVEL -> roll weapon pool [1 melee + 2 ranged]
            pool = roll_level_weapon_pool()
            base_weapon = Weapon.from_type("Пила")
            chosen_weapon = Weapon.from_type(pool[0])
            st = LevelState(level_idx=1, cart=Cart(), base_weapon=base_weapon, chosen_weapon=chosen_weapon, coins=BASE_COINS)
            st.level_weapon_pool = pool[:]
            st.init_level()
            gs.incoming_state = st
            self._level_snapshot = st
        else:
            gs.incoming_state = self._level_snapshot
        self.sm.add_widget(gs)
        self.sm.current = 'game'

    def start_game_new(self):
        self.clear_level_snapshot()
        self.start_game_continue()

if __name__ == '__main__':
    save = load_save(); save_save(save)
    TowerCartTDApp().run()
