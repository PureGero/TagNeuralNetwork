/** Create a neural network with a certain number of inputs and outputs */
function NeuralNetwork(inputs, outputs) {
    this.inputs = inputs;
    this.outputs = outputs;
    
    this.neurons = [];
    
    for (var i = 0; i < this.inputs + this.outputs + 1; i++) {
        this.neurons.push({});
    }
}

/** Get a raw neuron */
NeuralNetwork.prototype.getNeuron = function(i) {
    if (i in this.neurons) return this.neurons[i];
    console.error('getNeuron: ' + i + ' is not a valid neuron');
    return {};
}

/** Add a new neuron to the network and return its raw index */
NeuralNetwork.prototype.addNewNeuron = function() {
    this.neurons.push({});
    return this.neurons.length - 1;
}

/** Returns 1 + inputs + outputs + hiddens */
NeuralNetwork.prototype.size = function() {
    return this.neurons.length;
}

/** Get the value of a raw neuron */
NeuralNetwork.prototype.getValue = function(i) {
    var neuron = this.getNeuron(i);
    if ('value' in neuron) return neuron.value;
    return 0;
}

/** Set the value of a raw neuron */
NeuralNetwork.prototype.setValue = function(i, value) {
    var neuron = this.getNeuron(i);
    neuron.value = value;
}

/** Get the raw index from an input's index */
NeuralNetwork.prototype.getInputIndex = function(i) {
    if (i >= this.inputs) console.error(i, " is out of input size: ", this.inputs);
    return i + 1;
}

/** Get the raw index from an output's index */
NeuralNetwork.prototype.getOutputIndex = function(i) {
    if (i >= this.outputs) console.error(i, " is out of input size: ", this.outputs);
    return i + 1 + this.inputs;
}

/** Get the raw index from a hidden's index */
NeuralNetwork.prototype.getHiddenIndex = function(i) {
    return i + 1 + this.inputs + this.outputs;
}

/** Set the value of an input neuron */
NeuralNetwork.prototype.setInputValue = function(inputIndex, value) {
    return this.setValue(this.getInputIndex(inputIndex), value);
}

/** Get the value of an output neuron */
NeuralNetwork.prototype.getOutputValue = function(outputIndex) {
    return this.getValue(this.getOutputIndex(outputIndex));
}

/** Gets the list of {src, weight} inputs. Do not modify this list directly */
NeuralNetwork.prototype.getInputs = function(i) {
    var neuron = this.getNeuron(i);
    if ('inputs' in neuron) return neuron.inputs;
    return [];
}

/** Adds an input from a src, overwriting an existing one from the same src
    if it exists */
NeuralNetwork.prototype.addInput = function(i, src, weight) {
    var neuron = this.getNeuron(i);
    if (!('inputs' in neuron))
        neuron.inputs = [];
    
    for (var i = 0; i < neuron.inputs.length; i++) {
        if (neuron.inputs[i].src == src) {
            neuron.inputs[i].weight = weight;
            return;
        }
    }
    
    neuron.inputs.push({src: src, weight: weight});
}

/** Update a raw neuron from its inputs */
NeuralNetwork.prototype.updateNeuron = function(i) {
    var inputs = this.getInputs(i);
    var sum = 0;
    
    for (var j = 0; j < inputs.length; j++) {
        sum += this.getValue(inputs[j].src) * inputs[j].weight;
    }
    
    var value = sum > 0 ? 1 : 0;
    this.setValue(i, value);
}

/** Run the Neural Network, updating the values of the output neurons */
NeuralNetwork.prototype.run = function() {
    // Set biased neuron value
    this.setValue(0, 1);

    // Update all the hidden neurons
    for (var i = this.inputs + this.outputs + 1; i < this.size(); i++) {
        this.updateNeuron(i);
    }
    
    // Update the output neurons
    for (var i = this.inputs + 1; i < this.inputs + this.outputs + 1; i++) {
        this.updateNeuron(i);
    }
}

/** Reset all the neuron values to prepare them for a new instance */
NeuralNetwork.prototype.reset = function() {
    for (var i = 0; i < this.neurons.length; i++) {
        delete this.neurons[i].value;
    }
}

/** Dump all the neurons into a string for saving to a disk */
NeuralNetwork.prototype.dumpNeurons = function() {
    return JSON.stringify(this.neurons);
}

/** Load a string of neurons */
NeuralNetwork.prototype.loadNeurons = function(jsonString) {
    neurons = JSON.parse(jsonString);
}

/** Returns a clone of this network, recommend reset() before clone() */
NeuralNetwork.prototype.clone = function() {
    newNetwork = new NeuralNetwork(this.inputs, this.outputs);
    newNetwork.neurons = JSON.parse(JSON.stringify(this.neurons));
    return newNetwork;
}

/** Link two layers together */
NeuralNetwork.prototype.linkLayers = function(layer1, layer2) {
    for (var i = 0; i < layer1.length; i++) {
        for (var j = 0; j < layer2.length; j++) {
            this.addInput(layer2[j], layer1[i], 0);
            this.addInput(layer2[j], 0, 0); // Biased
        }
    }
}

/** Inputs, ...HiddenLayers, Outputs */
NeuralNetwork.createHiddenLayeredNetwork = function(...layerCounts) {
    neuralNetwork = new NeuralNetwork(layerCounts[0], layerCounts[layerCounts.length-1]);
    
    var layers = [];
    for (var j = 1; j < layerCounts.length - 1; j++) {
        var layer = [];
        for (var i = 0; i < layerCounts[j]; i++) {
            layer.push(neuralNetwork.addNewNeuron());
        }
        layers.push(layer);
    }
    
    var inputLayer = [];
    for (var i = 0; i < neuralNetwork.inputs; i++)
        inputLayer.push(neuralNetwork.getInputIndex(i));
        
    var outputLayer = [];
    for (var i = 0; i < neuralNetwork.outputs; i++)
        outputLayer.push(neuralNetwork.getOutputIndex(i));
        
    if (layers.length == 0) {
        neuralNetwork.linkLayers(inputLayer, outputLayer);
    } else {
        neuralNetwork.linkLayers(inputLayer, layers[0]);
        neuralNetwork.linkLayers(layers[layers.length - 1], outputLayer);
    }
    
    for (var i = 1; i < layers.length; i++)
        neuralNetwork.linkLayers(layers[i - 1], layers[i]);
        
    return neuralNetwork;
}

if ('module' in this) {
    module.exports = NeuralNetwork;
}