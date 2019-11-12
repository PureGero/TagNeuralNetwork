// Requires neuralnetwork.js

function Neat(species, parents, neuralNetwork) {
    this.species = species;
    this.parents = parents;
    this.evolveRate = 0.75;
    
    this.networks = [];
    this.scores = [];
    for (var i = 0; i < this.species; i++) {
        this.networks.push(neuralNetwork.clone());
        this.scores.push(0);
    }
    
    this.evolve();
}

Neat.prototype.breed = function() {
    var parents = [];

    for (var i = 0; i < this.networks.length; i++) {
        this.networks[i].reset();
        parents.push({network: this.networks[i], score: this.scores[i]});
    }

    parents.sort((a, b) => {
        // Order largest to smallest
        return ((a.score < b.score) ? 1 : ((a.score == b.score) ? 0 : -1));
    });
    
    this.scores = [];
    for (var i = 0; i < this.species; i++) this.scores.push(0);

    if (parents[0].score == parents[parents.length - 1].score) {
        console.log('Everyone has the same score, skipping breeding');
        return;
    }
    
    console.log('Best score is ' + parents[0].score);
    this.networks = [];

    for (var i = 0; i < this.parents; i++) {
        for (var j = 0; j < this.species/this.parents; j++) {
            this.networks.push(parents[i].network.clone());
        }
    }
}

Neat.prototype.evolve = function evolve() {
    for (var i = 0; i < this.networks.length; i++) {
        if ((i % (this.species/this.parents)) == 0) continue; // Don't evolve parents
        Neat.evolveHiddenLayeredNetwork(this.networks[i]);
    }
}

Neat.addRandomInput = function(network) {
    var dest = Math.floor(Math.random() * (network.size() - network.inputs - 1))
        + network.inputs + 1;
    
    var src = Math.floor(Math.random() * (network.size() - network.outputs));
    if (src > network.inputs + 1) src += network.outputs;
    
    var weight = Math.random()*2 - 1;
    
    network.addInput(dest, src, weight);
}

Neat.removeRandomInput = function(network) {
    var dest = Math.floor(Math.random() * (network.size() - network.inputs - 1))
        + network.inputs + 1;
    
    var inputs = network.getInputs(dest);
    
    inputs.splice(Math.floor(Math.random()*inputs.length), 1);
}

Neat.modifyRandomInput = function(network) {
    var dest = Math.floor(Math.random() * (network.size() - network.inputs - 1))
        + network.inputs + 1;
    
    var inputs = network.getInputs(dest);
    
    var input = inputs[Math.floor(Math.random()*inputs.length)];
    
    input.weight = Math.random()*2 - 1;
}

Neat.evolveNeuralNetwork = function(network) {
    do {
        if (Math.random() < 0.05) {
            Neat.addNewNeuron(network);
        }
        if (Math.random() < 0.5) {
            Neat.removeRandomInput(network);
        }
        Neat.addRandomInput(network);
    } while (Math.random() < this.evolveRate);
}

Neat.evolveHiddenLayeredNetwork = function(network) {
    do {
        Neat.modifyRandomInput(network);
    } while (Math.random() < this.evolveRate);
}

if (typeof module !== 'undefined') {
    module.exports = Neat;
}