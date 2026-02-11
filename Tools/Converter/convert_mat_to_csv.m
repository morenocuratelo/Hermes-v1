% =========================================================================
% SCRIPT: MAT to CSV Bulk Converter (GENERICO)
% STRUTTURA CARTELLE RICHIESTA:
%   ./ (root) -> contiene questo script
%   ./input/  -> inserisci qui QUALSIASI file .mat
%   ./output/ -> conterrà una cartella per ogni file .mat processato
%
% OBIETTIVO: Esportare tutte le variabili preservando i dati numerici
%            da tutti i file .mat presenti nella cartella input.
% =========================================================================

clear; clc;

% 1. Configurazione Percorsi
inputFolder = 'input';
outputFolder = 'output';

% 2. Verifiche Preliminari
if ~exist(inputFolder, 'dir')
    error('La cartella "input" non esiste. Creala e inserisci i file .mat al suo interno.');
end

% Trova tutti i file .mat nella cartella input
matFiles = dir(fullfile(inputFolder, '*.mat'));

if isempty(matFiles)
    error('Nessun file .mat trovato nella cartella "input".');
end

% Creazione cartella output principale se non esiste
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

fprintf('Trovati %d file .mat da processare.\n\n', length(matFiles));

% 3. Ciclo su ogni file trovato
for k = 1:length(matFiles)
    baseFileName = matFiles(k).name;
    sourceFile = fullfile(inputFolder, baseFileName);
    
    % Genera nome cartella output specifico per questo file (senza estensione)
    [~, nameNoExt, ~] = fileparts(baseFileName);
    currentOutputFolder = fullfile(outputFolder, nameNoExt);
    
    if ~exist(currentOutputFolder, 'dir')
        mkdir(currentOutputFolder);
    end
    
    fprintf('=== Elaborazione file %d/%d: %s ===\n', k, length(matFiles), baseFileName);
    
    try
        % Caricamento Dati
        dataStruct = load(sourceFile);
        varNames = fieldnames(dataStruct);
        fprintf('   Variabili trovate: %d\n', length(varNames));
        
        % Iterazione Variabili interne al singolo file
        for i = 1:length(varNames)
            currVarName = varNames{i};
            currData = dataStruct.(currVarName);
            
            % Definizione percorso file di output
            outputFile = fullfile(currentOutputFolder, [currVarName, '.csv']);
            
            fprintf('   -> Export %s: ', currVarName);
            
            try
                % Gestione Matrici Multidimensionali (>2D)
                if isnumeric(currData) && ndims(currData) > 2
                    sz = size(currData);
                    currData = reshape(currData, sz(1), []);
                    fprintf('[Reshaped %dD to 2D] ', length(sz));
                end
                
                % Gestione Structs
                if isstruct(currData)
                    currData = struct2table(currData, 'AsArray', true);
                    fprintf('[Struct to Table] ');
                end
                
                % Scrittura su Disco
                if istable(currData) || istimetable(currData)
                    writetable(currData, outputFile);
                    fprintf('OK\n');
                    
                elseif iscell(currData)
                    writecell(currData, outputFile);
                    fprintf('OK\n');
                    
                elseif isnumeric(currData) || islogical(currData)
                    writematrix(currData, outputFile);
                    fprintf('OK\n');
                    
                elseif ischar(currData) || isstring(currData)
                    fid = fopen(outputFile, 'wt');
                    if ischar(currData)
                        fprintf(fid, '%s', currData);
                    else
                        fprintf(fid, '%s\n', currData);
                    end
                    fclose(fid);
                    fprintf('OK (Text)\n');
                    
                else
                    warning('Skip: Tipo non supportato (%s)', class(currData));
                end
                
            catch ME
                fprintf('ERRORE (%s)\n', ME.message);
            end
        end
        fprintf('   Completato. Output in: %s\n\n', currentOutputFolder);
        
    catch ME
        fprintf('   ERRORE critico nel caricamento file: %s\n\n', ME.message);
    end
end

fprintf('Tutte le operazioni completate.\n');