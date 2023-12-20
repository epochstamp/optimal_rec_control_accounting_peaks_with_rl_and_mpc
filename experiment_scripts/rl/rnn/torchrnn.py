import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

activations = {
    "tanh": nn.functional.tanh,
    "relu": nn.functional.relu,
    "relu6": nn.functional.relu6,
    "elu": nn.functional.elu,
    "sigmoid": nn.functional.sigmoid
}
class TorchActorCriticLSTMModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        hiddens_fc_actor=[64],
        actor_activation="relu",
        hiddens_fc_critic=[256, 256],
        critic_activation="relu",
        hiddens_fc_actor_after=[],
        actor_activation_after="relu",
        hiddens_fc_critic_after=[],
        critic_activation_after="relu",
        lstm_state_size_actor=256,
        lstm_state_size_critic=256,
        initializer_actor=None,
        initializer_critic=None,
        **model_config_kwargs
    ):
        
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        train_initial_state = model_config_kwargs.get("train_initial_state", False)
        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self._train_initial_state = train_initial_state
        self.lstm_state_size_actor = model_config_kwargs.get("lstm_state_size_actor", lstm_state_size_actor)
        self.lstm_state_size_critic = model_config_kwargs.get("lstm_state_size_critic", lstm_state_size_critic)
        self.hiddens_fc_actor = model_config_kwargs.get("hiddens_fc_actor", hiddens_fc_actor)
        self.hiddens_fc_critic = model_config_kwargs.get("hiddens_fc_critic", hiddens_fc_critic)
        self._actor_activation = model_config_kwargs.get("actor_activation", actor_activation)
        self._critic_activation = model_config_kwargs.get("critic_activation", critic_activation)
        self.hiddens_fc_actor_after = model_config_kwargs.get("hiddens_fc_actor_after", hiddens_fc_actor_after)
        self.hiddens_fc_critic_after = model_config_kwargs.get("hiddens_fc_critic_after", hiddens_fc_critic_after)
        self._actor_activation_after = model_config_kwargs.get("actor_activation_after", actor_activation_after)
        self._critic_activation_after = model_config_kwargs.get("critic_activation_after", critic_activation_after)
        
        self.fc_actor = None
        self.fc_critic = None
        if len(self.hiddens_fc_actor) > 0:
            self.fc_actor = [nn.Linear(self.obs_size, self.hiddens_fc_actor[0]), activations[self._actor_activation]]
            for i, hidden_fc_actor in enumerate(self.hiddens_fc_actor):
                if i > 0:
                    self.fc_actor += [nn.Linear(self.hiddens_fc_actor[i-1], self.hiddens_fc_actor[i]), activations[self._actor_activation]]
                if initializer_actor is not None:
                    initializer_actor(self.fc_actor[-2].weight)

        if len(self.hiddens_fc_critic) > 0:
            self.fc_critic = [nn.Linear(self.obs_size, self.hiddens_fc_critic[0]), activations[self._critic_activation]]
            for i, hidden_fc_critic in enumerate(self.hiddens_fc_critic):
                if i > 0:
                    self.fc_critic += [nn.Linear(self.hiddens_fc_critic[i-1], self.hiddens_fc_critic[i]), activations[self._critic_activation]]
                if initializer_critic is not None:
                    initializer_critic(self.fc_critic[-2].weight)
        

        self.lstm_actor = nn.LSTM((self.hiddens_fc_actor[-1] if self.fc_actor is not None else self.obs_size), lstm_state_size_actor, batch_first=True)
        self.lstm_critic = nn.LSTM((self.hiddens_fc_critic[-1] if self.fc_critic is not None else self.obs_size), lstm_state_size_critic, batch_first=True)
        self.fc_actor_after = None
        self.fc_critic_after = None
        if len(self.hiddens_fc_actor_after) > 0:
            self.fc_actor_after = [nn.Linear(lstm_state_size_actor, self.hiddens_fc_actor_after[0]), activations[self._actor_activation_after]]
            for i, hidden_fc_actor_after in enumerate(self.hiddens_fc_actor_after):
                if i > 0:
                    self.fc_actor_after += [nn.Linear(self.hiddens_fc_actor_after[i-1], self.hiddens_fc_actor_after[i]), activations[self._actor_activation_after]]
                #initializer_actor(self.fc_actor_after[-2].weight)
        if len(self.hiddens_fc_critic_after) > 0:
            self.fc_critic_after = [nn.Linear(lstm_state_size_critic, self.hiddens_fc_critic_after[0]), activations[self._critic_activation_after]]
            for i, hidden_fc_critic_after in enumerate(self.hiddens_fc_critic_after):
                if i > 0:
                    self.fc_critic_after += [nn.Linear(self.hiddens_fc_critic_after[i-1], self.hiddens_fc_critic_after[i]), activations[self._critic_activation_after]]
                #initializer_critic(self.fc_critic_after[-2].weight)
        if self.fc_actor_after is not None:
            self.action_branch = nn.Linear(self.hiddens_fc_actor_after[-1], num_outputs)
        else:
            self.action_branch = nn.Linear(lstm_state_size_actor, num_outputs)
        if self.fc_critic_after is not None:
            self.critic_branch = nn.Linear(self.hiddens_fc_critic_after[-1], 1)
        else:
            self.critic_branch = nn.Linear(lstm_state_size_critic, 1)
        
        initializer_actor(self.action_branch.weight)
        
        initializer_critic(self.critic_branch.weight)
        # Holds the current "base" output (before logits layer).
        self._features_actor = None
        self._features_critic = None
        self._initializer_actor = initializer_actor
        self._initializer_critic = initializer_critic
        self._h = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        if not self._train_initial_state:
            h = [
                torch.zeros(self.lstm_state_size_actor),
                torch.zeros(self.lstm_state_size_actor),
                torch.zeros(self.lstm_state_size_critic),
                torch.zeros(self.lstm_state_size_critic)
            ]
        else:
            if self._h is None:
                self._h = [
                    torch.nn.Parameter(torch.normal(0, 1, [self.lstm_state_size_actor]), requires_grad=True),
                    torch.nn.Parameter(self.fc_actor[-2].weight.new(1, self.lstm_state_size_actor).squeeze(0).normal_(), requires_grad=True),
                    torch.nn.Parameter(self.fc_critic[-2].weight.new(1, self.lstm_state_size_critic).squeeze(0).normal_(), requires_grad=True),
                    torch.nn.Parameter(self.fc_critic[-2].weight.new(1, self.lstm_state_size_critic).squeeze(0).normal_(), requires_grad=True)
                ]
            h = [
                he.data for he in self._h
            ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features_critic is not None, "must call forward() first"
        x = self._features_critic
        if self.fc_critic_after is not None:
            for hidden_critic in self.fc_critic_after:
                x = hidden_critic(x)
        return torch.reshape(self.critic_branch(x), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x2 = inputs
        if self.fc_critic is not None:
            for hidden_critic_layer in self.fc_critic:
                x2 = hidden_critic_layer(x2)
        x = inputs
        if self.fc_actor is not None:
            for hidden_actor_layer in self.fc_actor:
                x = hidden_actor_layer(x)

        self._features_actor, [h1, c1] = self.lstm_actor(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        self._features_critic, [h2, c2] = self.lstm_critic(
            x2, [torch.unsqueeze(state[2], 0), torch.unsqueeze(state[3], 0)]
        )
        x = self._features_actor
        if self.fc_actor_after is not None:
            for hidden_fc_actor in self.fc_actor_after:
                x = hidden_fc_actor(x)
        action_out = self.action_branch(x)
        return action_out, [torch.squeeze(h1, 0), torch.squeeze(c1, 0), torch.squeeze(h2, 0), torch.squeeze(c2, 0)]
