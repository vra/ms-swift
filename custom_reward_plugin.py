import re
from swift.rewards import ORM, orms
from swift.rewards.orm import MathAccuracy


class ConditionalMathReward(ORM):
    """Conditional reward: format/step/structure bonuses only apply when answer is correct.

    This prevents reward hacking where the model optimizes for formatting
    without actually solving the problem correctly.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.accuracy_orm = MathAccuracy()

    def __call__(self, completions, solution, **kwargs):
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for acc, completion in zip(acc_rewards, completions):
            if acc < 1.0:
                # Wrong answer → zero reward, no matter how pretty the formatting
                rewards.append(0.0)
                continue

            # Correct answer → base 1.0 + optional bonuses
            reward = 1.0

            # Format bonus: has \boxed{}
            if re.search(r'\\boxed\{[^}]+\}', completion):
                reward += 0.2

            # Step bonus: contains reasoning-step markers
            step_markers = re.findall(
                r'(Step\s*\d+|step\s*\d+|First|Then|Next|Finally|Firstly|Secondly|Therefore|Thus)',
                completion
            )
            n_steps = len(step_markers)
            if n_steps >= 3:
                reward += 0.2
            elif n_steps >= 1:
                reward += 0.1

            # Structure bonus: multi-line reasoning with equations
            newline_count = completion.count('\n')
            eq_count = completion.count('=')
            if newline_count >= 3 and eq_count >= 2:
                reward += 0.1

            rewards.append(min(reward, 1.5))

        return rewards


orms['conditional_math'] = ConditionalMathReward
