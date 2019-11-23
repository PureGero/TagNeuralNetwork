const app = require('express')();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');

const NeuralNetwork = require('./neuralnetwork.js');
const Neat = require('./neat.js');

var CPU_TARGET = 50;

var speed = 0.5;
    
var dots = [];

var GAMES = 128;
var GAMES_PER_GENERATION = 100;
var PAST_GAMES = 80;
var GENERATIONS_PER_TRAINING = 25;
var SECONDS_PER_GAME = 5;
var PLAYER_COUNT = 2;
var TAGGER_COUNT = 1;
var SQRT2ON2 = 0.7071067811865475;

var ticks = 0;
var generations = 0;
var games = [];
var dt = 1/20;
var training = 0;
var moves = 0;
var times_trained = 0;

var writtenPlayers = [];
var previousBests = [];
var players = [];
var baseNetwork = NeuralNetwork.createMemoryNetwork(PLAYER_COUNT*2, 6, 6, 6, 4);
var lastScore = 0;
for (var i = 0; i < PLAYER_COUNT; i++) {
    players.push(new Neat(GAMES, 4, baseNetwork));
    players[i].evolveRate = 0.75;
    previousBests.push([]);
}

app.get('/', function(req, res){
    res.send('Server up and running :D');
});

io.on('connection', function(socket) {
    socket.emit('players', writtenPlayers);
    socket.emit('msg', (training?'Red':'Blue') + ' dot has been trained');
});

function random(seed) {
    var x = Math.sin(seed) * 100000;
    return x - Math.floor(x);
}

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
        
        var velocity = dt*speed*((up ^ down) & (left ^ right) ? SQRT2ON2 : 1);

        if (up) dot.y -= velocity;
        if (down) dot.y += velocity;
        if (left) dot.x -= velocity;
        if (right) dot.x += velocity;

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

function runGame(networks, seed) {
    ticks = 0;
    dots = [];
    for (var i = 0; i < networks.length; i++) {
        var x = random(seed + i/networks.length);
        var y = random(seed + i/networks.length + Math.sin(i));
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
    
    if (moves == 0) {
        ticks = SECONDS_PER_GAME;
    }
}

function runGeneration(training) {
    players[training].evolve();
    for (var j = 0; j < GAMES; j++) {
        for (var i = 0; i < GAMES_PER_GENERATION; i++) {
            var networks = [];
            for (var k = 0; k < players.length; k++) {
                if (i < PAST_GAMES && training != k) { // Fight previous agents
                    if (i < previousBests[k].length) {
                        networks.push(previousBests[k][i]);
                    } else {
                        networks.push(baseNetwork);
                    }
                } else { // Fight current agents
                    networks.push(players[k].networks[training == k ? j : 0]);
                }
            }
            
            runGame(networks, i + generations*GAMES_PER_GENERATION);
            
            if (dots[training].tagger) // Is tagger
                players[training].scores[j] += (SECONDS_PER_GAME - ticks)/SECONDS_PER_GAME;
            else // Is runner and hasn't been tagged
                players[training].scores[j] += ticks/SECONDS_PER_GAME;
        }
    }
    generations++;
    return players[training].breed();
}

function run() {
    var t = Date.now();

    var score = runGeneration(training);
    
    if (score == lastScore && score >= GAMES_PER_GENERATION) {
        generations += GENERATIONS_PER_TRAINING - (generations % (GENERATIONS_PER_TRAINING));
        score = 0;
    }
    
    lastScore = score;

    if (generations % (GENERATIONS_PER_TRAINING) == 0) {
        previousBests[training].push(players[training].networks[0]);
        if (previousBests[training].length > PAST_GAMES) // Remove a random previous best
            previousBests[training].splice(Math.floor(Math.random()*previousBests[training].length), 1);
    
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
            var network = NeuralNetwork.createMemoryNetwork(PLAYER_COUNT*2, 6, 6, 6, 4);
            network.neurons = loadPlayers[i];
            players.push(new Neat(GAMES, 4, network));
        }
    }
    
    run();
})

http.listen(1612, function(){
    console.log('Listening on *:1612');
});
