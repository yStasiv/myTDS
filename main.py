
# main.py — v6: smooth progress, stop 35cm before each tower/boss, 24 FPS
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.properties import StringProperty, ObjectProperty, NumericProperty, ListProperty, DictProperty

import json, os

from gamecore import (
    Weapon, WeaponKind, Cart, LevelState, simulate_tick, TOWER_ENEMY_HP_MULT, BASE_COINS
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

WORLD_LENGTH = 1000.0  # cm
TOWER_XS = [150.0, 300.0, 450.0, 700.0, 950.0]
BOSS_X = 1000.0
STOP_BEFORE = 35.0  # cm
CAMERA_RANGE = (-50.0, 50.0)  # show ~500cm window around the cart

# Precompute stop positions (nodes to stop at): start + (each tower - 35) + (boss - 35)
STOP_POSITIONS = [0.0] + [max(0.0, x - STOP_BEFORE) for x in TOWER_XS + [BOSS_X]]  # len=7 (nodes 0..6)

# ---------- Segmented progress bar ----------
class TopProgress(Widget):
    ratio = NumericProperty(0.0)  # 0..1
    segments = NumericProperty(6) # 6 segments between 7 stop nodes
    def on_size(self, *a): self._redraw()
    def on_pos(self, *a): self._redraw()
    def on_ratio(self, *a): self._redraw()
    def _redraw(self):
        self.canvas.before.clear()
        with self.canvas.before:
            # background bar
            Color(1,1,1,0.12)
            Rectangle(pos=self.pos, size=self.size)
            # segment ticks
            seg_w = self.width / float(self.segments)
            for i in range(1, self.segments):
                x = self.x + seg_w * i
                Color(1,1,1,0.24)
                Line(points=[x, self.y, x, self.top], width=1)
            # fill according to ratio
            Color(0.3,0.8,1.0,0.95)
            w = max(0.0, min(1.0, self.ratio)) * self.width
            Rectangle(pos=self.pos, size=(w, self.height))

# ---------- Playfield with camera ----------
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
            Color(0.10, 0.12, 0.16, 1)
            Rectangle(pos=self.pos, size=self.size)

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
        # Use stop positions (0, 115, 265, 415, 665, 915, 965)
        nodes = STOP_POSITIONS
        seg = max(0, min(state.segment_idx, len(nodes)-2))
        x0, x1 = nodes[seg], nodes[seg+1]
        # If paused — clamp to x0 to avoid any oscillations
        if phase != 'running' or state.paused_at_tower:
            return x0
        return x0 + (x1 - x0) * max(0.0, min(1.0, seg_progress))

    def render(self, state, seg_progress: float, dt: float, phase: str):
        self.canvas.after.clear()
        with self.canvas.after:
            cart_wx = self.cart_world_x(state, seg_progress, phase)
            cam_left, cam_right, _ = self._camera_window(cart_wx)

            # road
            Color(1,1,1,0.12)
            Line(points=[ self._w2s(cam_left, cam_left, cam_right), self.baseline_y,
                          self._w2s(cam_right, cam_left, cam_right), self.baseline_y ], width=1)

            # towers
            for i, wx in enumerate(self.towers_world_x):
                if not (cam_left <= wx <= cam_right): continue
                tower = next((t for t in state.towers if t.idx == i), None)
                if not tower: continue

                was_destroyed = self.destroyed_prev.get(i, False)
                if tower.destroyed and not was_destroyed:
                    self.destroy_flash[i] = 0.25
                self.destroyed_prev[i] = tower.destroyed

                sx = self._w2s(wx, cam_left, cam_right)
                if tower.destroyed:
                    Color(0.25,0.28,0.32,1)
                elif i == next((t.idx for t in state.towers if not t.destroyed), 4):
                    Color(0.85,0.45,0.25,1)
                else:
                    Color(0.55,0.62,0.72,1)
                Rectangle(pos=(sx - self.tower_w/2, self.baseline_y), size=(self.tower_w, self.tower_h))

                # HP bar
                maxhp = 300 * TOWER_ENEMY_HP_MULT[i]
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

            # boss
            if cam_left <= self.boss_world_x <= cam_right:
                sx = self._w2s(self.boss_world_x, cam_left, cam_right)
                Color(0.9,0.2,0.2,1)
                Rectangle(pos=(sx - self.tower_w/2, self.baseline_y + self.tower_h*0.15), size=(self.tower_w, self.tower_h*0.7))

            # enemies dots (for feel)
            # Active tower is the first alive; enemies between its tower and cart
            if any(not t.destroyed for t in state.towers):
                active_idx = next(t.idx for t in state.towers if not t.destroyed)
                tower_wx = self.towers_world_x[active_idx]
            else:
                tower_wx = self.towers_world_x[-1]

            n = min(10, len(state.enemies))
            for k in range(n):
                t = (k + 1) / (n + 1)
                ex = tower_wx + (cart_wx - tower_wx) * t
                if cam_left <= ex <= cam_right:
                    sx = self._w2s(ex, cam_left, cam_right)
                    Color(0.85,0.85,1.0,1); Ellipse(pos=(sx - 4, self.baseline_y - 4), size=(8, 8))

            # cart
            if cam_left <= cart_wx <= cam_right:
                cx = self._w2s(cart_wx, cam_left, cam_right); cy = self.baseline_y - 18
                Color(0.95,0.95,0.95,1); Rectangle(pos=(cx - 18, cy - 12), size=(36, 24))
                Color(0.3,0.3,0.3,1); Ellipse(pos=(cx - 16, cy - 18), size=(10, 10)); Ellipse(pos=(cx + 6, cy - 18), size=(10, 10))

            if phase == 'build':
                Color(1,1,1,0.05); Rectangle(pos=self.pos, size=self.size)

# ---------- GameRoot ----------
class GameRoot(BoxLayout):
    external_state = ObjectProperty(allownone=True, rebind=True)

    status_text = StringProperty("")
    phase = StringProperty("build")
    state = ObjectProperty(allownone=True, rebind=True)

    cart_cost_text = StringProperty("Апгр. Вагонетки")
    weapon_cost_text = StringProperty("Апгр. Зброї")
    progress_pct_text = StringProperty("0.0%")

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.seg_progress = 0.0    # 0..1 inside current segment
        self.time_scale = 1.0      # global time scale (x1/x2/x4)
        self.cart_speed_cm_s = 1.0 # absolute speed
        self._display_ratio = 0.0  # EMA for progress bar to avoid flicker
        # 24 FPS
        self._ev = Clock.schedule_interval(self.update, 1/24)
        self._ensure_state_initial()

    def on_external_state(self, *_):
        if self.external_state is not None:
            self.state = self.external_state
            self._enter_build_phase("Режим добудови: покращуйте вагонетку/зброю і натисніть 'Старт'")

    def _ensure_state_initial(self):
        if self.state is None:
            base_weapon = Weapon(name="Вагонетка", kind=WeaponKind.RANGED, base_damage=1.0, shots_per_second=3.0)
            chosen_weapon = Weapon(name="Арбалет", kind=WeaponKind.RANGED, base_damage=6.0, shots_per_second=0.5)
            self.state = LevelState(level_idx=1, cart=Cart(), base_weapon=base_weapon, chosen_weapon=chosen_weapon, coins=BASE_COINS)
            self.state.init_level()
        self._enter_build_phase("Режим добудови: покращуйте вагонетку/зброю і натисніть 'Старт'")

    def _enter_build_phase(self, msg):
        self.phase = 'build'
        self.seg_progress = 0.0
        self.status_text = msg
        self._refresh_prices_text()

    # --- pricing helpers (UI) ---
    def _cart_next_cost(self):
        lvl = getattr(self.state.cart, "level", 0)
        costs = [50,100,150,200]
        if lvl >= 4: return None
        return costs[lvl]

    def _weapon_next_cost(self):
        w = self.state.chosen_weapon
        k = getattr(w, "upgrades_done_in_tier", 0)
        base = 30.0
        if k < 5:
            return int(round(base * (1.15 ** k)))
        s = sum(int(round(base * (1.15 ** i))) for i in range(5))
        return s

    def _refresh_prices_text(self):
        c = self._cart_next_cost()
        self.cart_cost_text = f"Вагонетка +1 ({c} монет)" if c is not None else "Вагонетка MAX"
        w = self._weapon_next_cost()
        self.weapon_cost_text = f"Зброя + ({w} монет)" if w is not None else "Зброя MAX"

    # --- attempt reset (towers restored each try) ---
    def _reinit_for_attempt(self):
        coins = self.state.coins
        cart = self.state.cart
        base_w = self.state.base_weapon
        chosen_w = self.state.chosen_weapon
        lvl = self.state.level_idx
        self.state = LevelState(level_idx=lvl, cart=cart, base_weapon=base_w, chosen_weapon=chosen_w, coins=coins, crystals=self.state.crystals)
        self.state.init_level()
        self.state.segment_idx = 0
        self.state.paused_at_tower = False
        self.seg_progress = 0.0
        self._display_ratio = 0.0

    # --- UI actions ---
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
            if self.state.chosen_weapon.upgrades_done_in_tier == 5:
                bonus = "armor" if (self.state.chosen_weapon.tier % 2) else "regen"
                self.state.chosen_weapon.pick_bonus(bonus)
                self.status_text = f"Бонус за 5 покращень: {'броня' if bonus=='armor' else 'реген'}"
        self._refresh_prices_text()

    def cycle_speed(self):
        self.time_scale = {1.0: 2.0, 2.0: 4.0}.get(self.time_scale, 1.0)
        self.status_text = f"Швидкість гри x{int(self.time_scale)}"

    def _advance_segment_if_needed(self):
        # called when moving between nodes
        if self.seg_progress >= 1.0:
            self.seg_progress = 0.0
            self.state.segment_idx = min(self.state.segment_idx + 1, 6)  # last node index = 6
            if self.state.segment_idx <= 6:
                self.state.paused_at_tower = True

    def _maybe_unpause_if_cleared(self):
        if not self.state.paused_at_tower:
            return
        node = self.state.segment_idx  # 1..6
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
        # HUD
        self.ids.lbl_coins.text = f"Монети: {self.state.coins}"
        self.ids.lbl_hp.text = f"HP: {self.state.cart.hp}/{self.state.cart.max_hp}"
        self.ids.lbl_level.text = f"Рівень: {self.state.level_idx}"
        self.ids.lbl_segment.text = f"Сегмент: {self.state.segment_idx}"

        self._refresh_prices_text()

        # Smooth progress percent (EMA)
        cart_wx = self.ids.playfield.cart_world_x(self.state, self.seg_progress, self.phase)
        instant_ratio = 1.0 if (self.state.segment_idx == 6 and not self.state.boss_alive) else max(0.0, min(1.0, cart_wx / self.ids.playfield.world_len))
        # Exponential moving average to prevent flicker
        alpha = 0.25  # smoothing factor per frame at 24 FPS
        self._display_ratio = self._display_ratio + (instant_ratio - self._display_ratio) * alpha
        self.ids.top_progress.ratio = self._display_ratio
        self.progress_pct_text = f"{self._display_ratio*100:.1f}%"

        if self.phase == 'running':
            scaled_dt = dt * self.time_scale
            simulate_tick(self.state, scaled_dt)

            # movement using stop positions
            nodes = STOP_POSITIONS
            seg = max(0, min(self.state.segment_idx, len(nodes)-2))
            seg_len = max(1e-6, (nodes[seg+1] - nodes[seg]))  # cm between stops
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

            # victory when boss dead and we're at node 6 (stop before boss)
            if not self.state.boss_alive and self.state.segment_idx >= 6:
                self._on_victory(); return

        # scaled dt for visual effects
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

# --- Screens & App ---
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
            base_weapon = Weapon(name="Вагонетка", kind=WeaponKind.RANGED, base_damage=1.0, shots_per_second=3.0)
            chosen_weapon = Weapon(name="Арбалет", kind=WeaponKind.RANGED, base_damage=6.0, shots_per_second=0.5)
            st = LevelState(level_idx=1, cart=Cart(), base_weapon=base_weapon, chosen_weapon=chosen_weapon, coins=BASE_COINS)
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
    save_save(load_save())
    TowerCartTDApp().run()
