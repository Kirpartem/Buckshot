import numpy as np
from core.constants import *


class Player:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.hp: int = 4
        self.max_inventory_capacity: int = 8
        self.handcuff_strength = 0
        self.items: list[Item] = []
        self.known_next: bool = False

    def get_items(self, num_items: int = 2) -> None:
        for _ in range(num_items):
            if len(self.items) >= self.max_inventory_capacity:
                return
            self.items.append(self.rng.choice(ITEMS))


class BuckshotRouletteGame:
    def __init__(self, rng_seed: int = 0):
        self.rng = np.random.default_rng(rng_seed)
        self.round: int = 1
        self.sub_round: int = 1
        self.turn: Turn = self.rng.choice([Turn.PLAYER, Turn.DEALER])
        self.player: Player = Player(self.rng)
        self.dealer: Player = Player(self.rng)
        self.saw_active: bool = False
        self.max_hp = 5
        self.max_bullets = 8
        self.bullet_sequence: list[int] = self.generate_bullet_sequence()
        

    def generate_bullet_sequence(self, num_lives: int = 1, num_blanks: int = 1) -> list[int]:
        seq = [0] * num_blanks + [1] * num_lives
        return self.rng.permutation(seq).tolist()

    def clear_items(self) -> None:
        self.player.items = []
        self.dealer.items = []
    
    def give_items(self, num: int) -> None:
        self.player.get_items(num)
        self.dealer.get_items(num)

    def process_shooting_self(self, target: Player):
        damage = 2 if self.saw_active else 1
        bullet = self.bullet_sequence.pop(0)
        if bullet:
            target.hp = max(target.hp - damage, 0)
            self.switch_turns()
    
    def process_shooting_target(self, target: Player):
        damage = 2 if self.saw_active else 1
        bullet = self.bullet_sequence.pop(0)
        if bullet:
            target.hp = max(target.hp - damage, 0)
        self.switch_turns()
    
    def unhandcuff_both(self):
        self.player.handcuff_strength = 0
        self.dealer.handcuff_strength = 0
        
    def switch_turns(self):  # Stupid fucking logic
        if self.turn == Turn.PLAYER:
            if self.dealer.handcuff_strength < 2:
                self.turn = Turn.DEALER
                self.dealer.handcuff_strength = 0
            else:
                self.turn = Turn.PLAYER
                self.dealer.handcuff_strength = 1        
        elif self.turn == Turn.DEALER:
            if self.player.handcuff_strength < 2:
                self.turn = Turn.PLAYER
                self.player.handcuff_strength = 0
            else:
                self.turn = Turn.DEALER
                self.player.handcuff_strength = 1
    
    def clear_known_bullets(self):
        self.player.known_next = False
        self.dealer.known_next = False
    
    def start_new_subround(self):
        self.sub_round += 1
        sub_config = self._generate_combo()

        self.unhandcuff_both()
        self.give_items(sub_config.num_items)
        self.turn = self.rng.choice([Turn.PLAYER, Turn.DEALER])
        self.bullet_sequence = self.generate_bullet_sequence(sub_config.lives, sub_config.blanks)
        self.clear_known_bullets()
    
    def _get_opponent(self, player: Player) -> Player:
        return self.dealer if player is self.player else self.player
    
    def _generate_combo(self) -> SubroundCombo:
        match self.sub_round:
            case 1:
                hp = self.rng.integers(4, 6)
                num_bullets = self.rng.integers(2, 5)
                lives_percentage = self.rng.uniform(0.25, 0.5)
                num_items = self.rng.integers(0, 2)  # [0, 1]
            case 2:
                hp = self.rng.integers(4, 6)
                num_bullets = self.rng.integers(2, 7)
                lives_percentage = self.rng.uniform(0.3, 0.6)
                num_items = self.rng.integers(1, 3)  # [1, 2]
            case _:
                hp = self.rng.integers(3, self.max_hp)
                num_bullets = self.rng.integers(3, self.max_bullets)
                lives_percentage = self.rng.uniform(0.4, 0.8)
                probs = [
                    0.5, # For 1 item
                    0.35, # For 2 items
                    0.15 # For 3 items
                ]
                num_items = self.rng.choice([1, 2, 3], p=probs)
            
        lives = int(np.floor(num_bullets * lives_percentage))
        lives = max(1, min(lives, num_bullets - 1))
        blanks = num_bullets - lives
        return SubroundCombo(num_items, hp, blanks, lives)

    def start_new_round(self):
        self.round += 1
        self.sub_round = 0
        self.clear_items()

        self.round_config = self._generate_combo()
        self.player.hp = self.round_config.starting_hp
        self.dealer.hp = self.round_config.starting_hp
        self.start_new_subround()
        
    def prepare_for_next_turn(self):
        if len(self.bullet_sequence) == 0: 
            self.start_new_subround()

    def process_action_result(self, action: GameAction):
        initiator, target = self.get_current_actor()
            
        match action:
            case GameAction.SHOOT_TARGET:
                self.process_shooting_target(target)
                self.saw_active = False
                initiator.known_next = False
            case GameAction.SHOOT_SELF:
                self.process_shooting_self(initiator)
                self.saw_active = False  # Force disable saw
                initiator.known_next = False
            case GameAction.USE_GLASS:
                initiator.known_next = True
                initiator.items.remove(Item.GLASS)
            case GameAction.USE_CIGARETTES:
                if initiator.hp < self.max_hp:
                    initiator.hp += 1 
                initiator.items.remove(Item.CIGARETTES)
            case GameAction.USE_HANDCUFFS:
                target.handcuff_strength = 2 
                initiator.items.remove(Item.HANDCUFFS)
            case GameAction.USE_SAW:
                self.saw_active = True
                initiator.items.remove(Item.SAW)
            case GameAction.USE_BEER:
                self.bullet_sequence.pop(0)
                self.clear_known_bullets()
                initiator.items.remove(Item.BEER)
        
        self.prepare_for_next_turn()
    
    def get_current_actor(self) -> tuple[Player, Player]:
        if self.turn == Turn.PLAYER:
            initiator, target = self.player, self.dealer
        else:
            initiator, target = self.dealer, self.player
        return initiator, target

    def get_valid_actions_mask(self) -> np.ndarray:
        """
        Efficiently generates the action mask without any loops.
        Returns a mask where 1 = valid action, 0 = invalid.
        """
        mask = np.zeros(len(GAME_ACTIONS), dtype=np.int8)
        actor, target = self.get_current_actor()
        items = actor.items

        # Always-valid actions
        mask[ACTION_MAP_INV[GameAction.SHOOT_SELF]] = 1
        mask[ACTION_MAP_INV[GameAction.SHOOT_TARGET]] = 1

        # Item-based actions (fast 'in' check)
        if Item.BEER in items:
            mask[ACTION_MAP_INV[GameAction.USE_BEER]] = 1

        if Item.CIGARETTES in items:
            mask[ACTION_MAP_INV[GameAction.USE_CIGARETTES]] = 1

        if Item.GLASS in items:
            mask[ACTION_MAP_INV[GameAction.USE_GLASS]] = 1

        if Item.HANDCUFFS in items and target.handcuff_strength == 0:
            mask[ACTION_MAP_INV[GameAction.USE_HANDCUFFS]] = 1

        if Item.SAW in items and not self.saw_active:
            mask[ACTION_MAP_INV[GameAction.USE_SAW]] = 1

        return mask

    def check_action_valid(self, player: Player, action: GameAction) -> bool:
        match action:
            case (GameAction.SHOOT_SELF | GameAction.SHOOT_TARGET):
                return True
            case GameAction.USE_BEER:
                return Item.BEER in player.items
            case GameAction.USE_CIGARETTES:
                return Item.CIGARETTES in player.items
            case GameAction.USE_GLASS:
                return Item.GLASS in player.items
            case GameAction.USE_HANDCUFFS:
                return Item.HANDCUFFS in player.items and self._get_opponent(player).handcuff_strength == 0
            case GameAction.USE_SAW:
                return Item.SAW in player.items and not self.saw_active
            case _:
                return False
            
    def step(self, action: GameAction) -> StepResult:
        """Apply an action and return structured step information."""
        initiator, target = self.get_current_actor()

        prev_bot_hp, prev_target_hp = initiator.hp, target.hp
        valid = self.check_action_valid(initiator, action)

        if valid:
            self.process_action_result(action)

        new_bot_hp, new_target_hp = initiator.hp, target.hp
        player_dead = self.player.hp <= 0
        dealer_dead = self.dealer.hp <= 0
        terminated = player_dead or dealer_dead

        info = {
            "turn": self.turn.name,
            "invalid_action": not valid,
        }

        return StepResult(
            valid=valid,
            action=action,
            prev_bot_hp=prev_bot_hp,
            prev_target_hp=prev_target_hp,
            new_bot_hp=new_bot_hp,
            new_target_hp=new_target_hp,
            player_dead=player_dead,
            dealer_dead=dealer_dead,
            terminated=terminated,
            info=info,
        )