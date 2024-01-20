
module julia_utils
include("../utils/julia_utils.jl")
end

using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--year", "-y"
            help = "Year of the mobility measurements"
            arg_type = Int
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
println("Retrieved args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end

julia_utils.complete_df2(parsed_args["year"])
