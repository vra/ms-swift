import re
from typing import List
from swift.rewards import ORM, orms
from swift.rewards.orm import MathAccuracy


class EnhancedMathReward(ORM):
    """自定义奖励函数：结合正确性和格式规范"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.accuracy_orm = MathAccuracy(args, **kwargs)

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        # 1. 正确性奖励（核心）
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        
        rewards = []
        for acc_reward, completion in zip(acc_rewards, completions):
            reward = acc_reward
            
            # 2. 格式奖励：如果使用了 \boxed{} 格式
            if re.search(r'\\boxed\{[^}]+\}', completion):
                reward += 0.05
            
            # 3. 步骤奖励：鼓励展示推理过程
            steps = completion.count('\n')
            if steps >= 3:
                reward += 0.05
            
            rewards.append(min(reward, 1.1))
        return rewards


orms['enhanced_math'] = EnhancedMathReward
