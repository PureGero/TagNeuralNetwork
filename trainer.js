const app = require('express')();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');

const NeuralNetwork = require('./neuralnetwork.js');
const Neat = require('./neat.js');

var CPU_TARGET = 50;

var speed = 0.5;
    
var dots = [];

var GAMES = 256;
var GAMES_PER_GENERATION = 50;
var GENERATIONS_PER_TRAINING = 50;
var SECONDS_PER_GAME = 5;
var PLAYER_COUNT = 2;
var TAGGER_COUNT = 1;

var ticks = 0;
var generations = 0;
var games = [];
var dt = 1/20;
var training = 0;
var moves = 0;

var writtenPlayers = [];
var previousBest3 = [];
var previousBest2 = [];
var previousBest = [];
var players = [];
var baseNetwork = NeuralNetwork.createHiddenLayeredNetwork(PLAYER_COUNT*2, 4, 4);
var lastScore = 0;
for (var i = 0; i < PLAYER_COUNT; i++) {
    players.push(new Neat(GAMES, 4, baseNetwork));
    players[i].evolveRate = 0.5;
    previousBest.push(baseNetwork);
    previousBest2.push(baseNetwork);
    previousBest3.push(baseNetwork);
}

app.get('/', function(req, res){
    res.send('Server up and running :D');
});

io.on('connection', function(socket) {
    socket.emit('players', writtenPlayers);
    socket.emit('msg', (training?'Red':'Blue') + ' dot has been trained');
});

function runDot(network, i) {
    var dot = dots[i];
    if (!dot.tagged) {

        for (var j = 0; j < dots.length; j++) {
            network.setInputValue(j*2, dots[j].x);
            network.setInputValue(j*2 + 1, dots[j].y);
            //network.setInputValue(j*3 + 2, dots[j].tagged ? 1 : 0);
        }

        network.run();

        var up = network.getOutputValue(0);
        var down = network.getOutputValue(1);
        var left = network.getOutputValue(2);
        var right = network.getOutputValue(3);
        
        var lx = dot.x;
        var ly = dot.y;
        
        if (up) dot.y -= speed*dt;
        if (down) dot.y += speed*dt;
        if (left) dot.x -= speed*dt;
        if (right) dot.x += speed*dt;

        if (dot.x < 0) dot.x = 0;
        if (dot.x > 1) dot.x = 1;
        if (dot.y < 0) dot.y = 0;
        if (dot.y > 1) dot.y = 1;
        
        if (dot.x != lx || dot.y != ly) moves += 1;
    
        if (dot.tagger) {
            for (var j = 0; j < dots.length; j++) {
                var d2 = dots[j];
                if (!d2.tagger && !d2.tagged && 
                        Math.abs(dot.x - d2.x) < 0.05 && 
                        Math.abs(dot.y - d2.y) < 0.05) {
                    d2.tagged = true;
                    dot.tagged_count += 1;
                }
            }
        }
    }
}

function runGame(networks) {
    ticks = 0;
    dots = [];
    for (var i = 0; i < networks.length; i++) {
        var x = Math.random();
        var y = Math.random();
        dots.push({
            x: x, 
            y: y, 
            tagged: false, 
            tagged_count: 0,
            tagger: i < TAGGER_COUNT
        });
    }
    
    while ((ticks+=dt) < SECONDS_PER_GAME) {
        moves = 0;
        
        for (var i = 0; i < dots.length; i++) {
            runDot(networks[i], i);
        }
        
        if (moves == 0 || dots[1].tagged) {
            break; // Game over
        }
    }
}

function runGeneration(training) {
    players[training].evolve();
    for (var j = 0; j < GAMES; j++) {
        for (var i = 0; i < GAMES_PER_GENERATION; i++) {
            var networks = [];
            for (var k = 0; k < players.length; k++) {
                if (i < GAMES_PER_GENERATION/2 || training == k) { // Fight current agents
                    networks.push(players[k].networks[training == k ? j : 0]);
                } else if (i < GAMES_PER_GENERATION*4/6) { // Fight previous agents
                    networks.push(previousBest[k]);
                } else if (i < GAMES_PER_GENERATION*5/6) { // Fight previous agents
                    networks.push(previousBest2[k]);
                } else { // Fight previous agents
                    networks.push(previousBest3[k]);
                }
            }
            
            runGame(networks);
            
            if (dots[training].tagger) // Is tagger
                players[training].scores[j] += dots[training].tagged_count;
            else if (!dots[training].tagged) // Is runner and hasn't been tagged
                players[training].scores[j] += 1;
        }
    }
    generations++;
    return players[training].breed();
}

function run() {
    var t = Date.now();

    var score = runGeneration(training);
    
    if (score == lastScore && score == GAMES_PER_GENERATION) {
        generations += GENERATIONS_PER_TRAINING - (generations % (GENERATIONS_PER_TRAINING));
        score = 0;
    }
    
    lastScore = score;

    if (generations % (GENERATIONS_PER_TRAINING) == 0) {
        previousBest2[training] = previousBest[training];
        previousBest[training] = players[training].networks[0];
    
        training = (training + 1) % PLAYER_COUNT;
        console.log("Begin training of", training);
        io.emit('msg', (training?'Red':'Blue') + ' dot has been trained');
        writeBestToDisk();
    }
    
    var dt = Date.now() - t;
    setTimeout(run, dt/(CPU_TARGET/100)*(1-CPU_TARGET/100));
}

function writeBestToDisk() {
    writtenPlayers = [];
    for (var i = 0; i < players.length; i++) {
        players[i].networks[0].reset();
        writtenPlayers.push(players[i].networks[0].neurons);
    }
    fs.writeFile("players.json", JSON.stringify(writtenPlayers), (err) => {
        if (err) console.log(err);
    });
    
    io.emit('players', writtenPlayers);
}

fs.readFile("players.json", "utf-8", (err, data) => {
    if (err) { 
        console.log(err);
    } else {
        var loadPlayers = JSON.parse(data);
        players = [];
        for (var i = 0; i < loadPlayers.length; i++) {
            var network = NeuralNetwork.createHiddenLayeredNetwork(PLAYER_COUNT*2, 4, 4);
            network.neurons = loadPlayers[i];
            players.push(new Neat(GAMES, 4, network));
        }
    }
    
    run();
})

http.listen(1612, function(){
    console.log('Listening on *:1612');
});